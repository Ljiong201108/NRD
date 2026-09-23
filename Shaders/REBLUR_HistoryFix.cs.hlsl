/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NRD.hlsli"
#include "ml.hlsli"

#include "REBLUR_Config.hlsli"
#include "REBLUR_HistoryFix.resources.hlsli"

#include "Common.hlsli"

#include "REBLUR_Common.hlsli"

groupshared float s_DiffLuma[BUFFER_Y][BUFFER_X];
groupshared float s_SpecLuma[BUFFER_Y][BUFFER_X];
groupshared float2 s_FrameNum[BUFFER_Y][BUFFER_X];
groupshared float4 s_Normal_Material[BUFFER_Y][BUFFER_X];
groupshared float s_ViewZ[BUFFER_Y][BUFFER_X];

void Preload(uint2 sharedPos, int2 globalPos) {
    globalPos = clamp(globalPos, 0, gRectSizeMinusOne);

#if (NRD_DIFF)
    s_DiffLuma[sharedPos.y][sharedPos.x] = gIn_DiffFast[globalPos];
#endif

#if (NRD_SPEC)
    s_SpecLuma[sharedPos.y][sharedPos.x] = gIn_SpecFast[globalPos];
#endif

    s_FrameNum[sharedPos.y][sharedPos.x] = UnpackData1(gIn_Data1[globalPos]);
    float materialID;
    float3 normal = NRD_FrontEnd_UnpackNormalAndRoughness(gIn_Normal_Roughness[WithRectOrigin(globalPos)], materialID).xyz;
    s_Normal_Material[sharedPos.y][sharedPos.x] = float4(normal, materialID);
    s_ViewZ[sharedPos.y][sharedPos.x] = UnpackViewZ(gIn_ViewZ[WithRectOrigin(globalPos)]);
}

// Tests 20, 23, 24, 27, 28, 54, 59, 65, 66, 76, 81, 98, 112, 117, 124, 126, 128, 134
// TODO: potentially do color clamping after reconstruction in a separate pass

float GetSurfaceWeight(int2 sharedPos, int2 pixelPos, float3 Xv, float3 Nv, float3 N, float materialID, float minMaterial) {
    float sampleViewZ = s_ViewZ[sharedPos.y][sharedPos.x];
    float4 guide = s_Normal_Material[sharedPos.y][sharedPos.x];
    pixelPos = clamp(pixelPos, 0, gRectSizeMinusOne);
    float3 sampleXv = Geometry::ReconstructViewPosition((pixelPos + 0.5) * gRectSizeInv, gFrustum, sampleViewZ, gOrthoMode);
    return float(sampleViewZ < gDenoisingRange && abs(dot(Nv, sampleXv - Xv)) <= max(NRD_DISOCCLUSION_THRESHOLD * abs(Xv.z), NRD_EPS)
        && dot(N, guide.xyz) >= 0.5) * CompareMaterials(materialID, guide.w, minMaterial);
}

[numthreads(GROUP_X, GROUP_Y, 1)] NRD_EXPORT void NRD_CS_MAIN(NRD_CS_MAIN_ARGS) {
    NRD_CTA_ORDER_REVERSED;

    // Preload
    float isSky = gIn_Tiles[pixelPos >> 4].x;
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // Tile-based early out
    if (isSky != 0.0 || any(pixelPos > gRectSizeMinusOne))
        return;

    // Early out
    float viewZ = UnpackViewZ(gIn_ViewZ[WithRectOrigin(pixelPos)]);
    if (viewZ > gDenoisingRange)
        return;

    // Center data
    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness(gIn_Normal_Roughness[WithRectOrigin(pixelPos)], materialID);
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    float frustumSize = GetFrustumSize(gMinRectDimMulUnproject, gOrthoMode, viewZ);
    float2 pixelUv = float2(pixelPos + 0.5) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition(pixelUv, gFrustum, viewZ, gOrthoMode);
    float3 Nv = Geometry::RotateVectorInverse(gViewToWorld, N);
    float3 Vv = GetViewVector(Xv, true);
    float NoV = abs(dot(Nv, Vv));

    int2 smemPos = threadPos + BORDER;
    float2 frameNum = s_FrameNum[smemPos.y][smemPos.x];

    float invHistoryFixFrameNum = 1.0 / max(gHistoryFixFrameNum, NRD_EPS);
    float2 frameNumAvgNorm = saturate(frameNum * invHistoryFixFrameNum);

    // Use smaller strides if neighborhood pixels have longer history to minimize chances of ruining contact details
    float baseStride = materialID == gHistoryFixAlternatePixelStrideMaterialID ? gHistoryFixAlternatePixelStride : gHistoryFixBasePixelStride;
    baseStride /= 1.0 + 1.0; // to match RELAX, where "frameNum" after "TemporalAccumulation" is "1", not "0"

    float2 stride = 1.0;

    [unroll] for (i = -1; i <= 1; i++) {
        [unroll] for (j = -1; j <= 1; j++) {
            if (i == 0 && j == 0)
                continue;

            float2 f = s_FrameNum[smemPos.y + j][smemPos.x + i];

            stride -= saturate((f - frameNum) * invHistoryFixFrameNum) / 9.0;
        }
    }

    stride = lerp(1.0, baseStride, stride);
    stride *= 2.0 / REBLUR_HISTORY_FIX_FILTER_RADIUS; // preserve blur radius in pixels ( default blur radius is 2 taps )
    stride *= float2(frameNum < gHistoryFixFrameNum);

// Diffuse
#if (NRD_DIFF)
    {
        REBLUR_TYPE diff = gIn_Diff[pixelPos];
#    if (NRD_MODE == SH)
        float4 diffSh = gIn_DiffSh[pixelPos];
#    endif

        float diffNonLinearAccumSpeed = 1.0 / (1.0 + frameNum.x);

        float hitDistScale = _REBLUR_GetHitDistanceNormalization(viewZ, gHitDistParams, 1.0);
        float hitDist = ExtractHitDist(diff) * hitDistScale;
        float hitDistFactor = GetHitDistFactor(hitDist, frustumSize);
        hitDist = ExtractHitDist(diff);

        // Stride between taps
        bool smoothOpaqueDiffuse = gEnableHalfRateTransmission != 0 && materialID < 0.5 && roughness <= 0.12;
        float diffStride = stride.x;
        if (!smoothOpaqueDiffuse) diffStride *= lerp(0.25 + 0.75 * Math::Sqrt01(hitDistFactor), 1.0, diffNonLinearAccumSpeed); // "hitDistFactor" is very noisy and breaks nice patterns
        diffStride = round(diffStride);
        bool smoothTransmission = gEnableHalfRateTransmission != 0 && materialID > 0.5 && roughness < 0.12;
        if (smoothTransmission)
            diffStride = min(diffStride, 1.0);

        // History reconstruction
        if (diffStride != 0.0) {
            // Parameters
            float normalWeightParam = GetNormalWeightParam(diffNonLinearAccumSpeed, gLobeAngleFraction);
            float2 geometryWeightParams = GetGeometryWeightParams(gPlaneDistSensitivity, frustumSize, Xv, Nv, diffNonLinearAccumSpeed);
            float hitDistWeightNorm = 1.0 / (0.5 * diffNonLinearAccumSpeed);

            float sumd = 1.0 + frameNum.x;
#    if (REBLUR_PERFORMANCE_MODE == 1)
            sumd = 1.0 + 1.0 / (1.0 + gMaxAccumulatedFrameNum) - diffNonLinearAccumSpeed;
#    endif

            diff *= sumd;
#    if (NRD_MODE == SH)
            diffSh *= sumd;
#    endif

            [unroll] for (j = -REBLUR_HISTORY_FIX_FILTER_RADIUS; j <= REBLUR_HISTORY_FIX_FILTER_RADIUS; j++) {
                [unroll] for (i = -REBLUR_HISTORY_FIX_FILTER_RADIUS; i <= REBLUR_HISTORY_FIX_FILTER_RADIUS; i++) {
                    // Skip center
                    if (i == 0 && j == 0)
                        continue;

                    // Skip corners
                    if (abs(i) + abs(j) == REBLUR_HISTORY_FIX_FILTER_RADIUS * 2)
                        continue;

                    // Sample uv ( at the pixel center )
                    float2 uv = pixelUv + float2(i, j) * diffStride * gRectSizeInv;

                    // Apply "mirror" to not waste taps going outside of the screen
                    uv = MirrorUv(uv);

                    // "uv" to "pos"
                    int2 pos = uv * gRectSize; // "uv" can't be "1"

                    // Fetch data
                    float zs = UnpackViewZ(gIn_ViewZ[WithRectOrigin(pos)]);
                    float3 Xvs = Geometry::ReconstructViewPosition(uv, gFrustum, zs, gOrthoMode);

                    float materialIDs;
                    float4 Ns = gIn_Normal_Roughness[WithRectOrigin(pos)];
                    Ns = NRD_FrontEnd_UnpackNormalAndRoughness(Ns, materialIDs);

                    // Weight
                    float angle = Math::AcosApprox(dot(Ns.xyz, N));
                    float NoX = dot(Nv, Xvs);

                    float w = ComputeWeight(NoX, geometryWeightParams.x, geometryWeightParams.y);
                    w *= CompareMaterials(materialID, materialIDs, gDiffMinMaterial);
                    w *= ComputeExponentialWeight(angle, normalWeightParam, 0.0);
                    w = zs < gDenoisingRange && dot(N, Ns.xyz) >= 0.5 ? w : 0.0; // |NoX| can be ~0 if "zs" is out of range
                                                        // gaussian weight is not needed

#    if (REBLUR_PERFORMANCE_MODE == 0)
                    w *= 1.0 + UnpackData1(gIn_Data1[pos]).x;
#    endif

                    REBLUR_TYPE s = gIn_Diff[pos];
                    s = Denanify(w, s);

                    // A-trous weight
                    float hs = ExtractHitDist(s);
                    float d = hs - hitDist; // use normalized hit distanced for simplicity ( no difference )
                    if (!smoothOpaqueDiffuse) w *= exp(-d * d * hitDistWeightNorm);

                    // Accumulate
                    sumd += w;

                    diff += s * w;
#    if (NRD_MODE == SH)
                    float4 sh = gIn_DiffSh[pos];
                    sh = Denanify(w, sh);
                    diffSh += sh * w;
#    endif
                }
            }

            sumd = Math::PositiveRcp(sumd);
            diff *= sumd;
#    if (NRD_MODE == SH)
            diffSh *= sumd;
#    endif
        }

        // Local variance
        float diffCenter = s_DiffLuma[threadPos.y + BORDER][threadPos.x + BORDER];
        float diffM1 = diffCenter;
        float diffM2 = diffM1 * diffM1;
        float diffWeight = 1.0;

        float f = frameNumAvgNorm.x;
        diffCenter = lerp(GetLuma(diff), diffCenter, f);
        gOut_DiffFast[pixelPos] = diffCenter;

        [unroll] for (j = 0; j <= BORDER * 2; j++) {
            [unroll] for (i = 0; i <= BORDER * 2; i++) {
                // Skip center
                if (i == BORDER && j == BORDER)
                    continue;

                int2 pos = threadPos + int2(i, j);

                float d = s_DiffLuma[pos.y][pos.x];
                float w = GetSurfaceWeight(pos, int2(pixelPos) + int2(i, j) - BORDER, Xv, Nv, N, materialID, gDiffMinMaterial);
                diffM1 += d * w;
                diffM2 += d * d * w;
                diffWeight += w;
            }
        }

        float diffLuma = GetLuma(diff);

        // Anti-firefly. The wide history ring is retained for transparent and
        // low-roughness paths. Rough opaque diffuse uses the already resident
        // 5x5 fast-history tile below: this catches isolated and two-pixel
        // crawlers without comparing a newly illuminated region against a
        // distant dark ring (which produced the moving-light black holes).
        bool roughOpaqueDiffuse = materialID < 0.5 && roughness > 0.12;
        if (gAntiFirefly && NRD_SUPPORTS_ANTIFIREFLY == 1 && !roughOpaqueDiffuse && !smoothTransmission) {
            float m1 = 0;
            float m2 = 0;

            [unroll] for (j = -REBLUR_ANTI_FIREFLY_FILTER_RADIUS; j <= REBLUR_ANTI_FIREFLY_FILTER_RADIUS; j++) {
                [unroll] for (i = -REBLUR_ANTI_FIREFLY_FILTER_RADIUS; i <= REBLUR_ANTI_FIREFLY_FILTER_RADIUS; i++) {
                    // Skip central 3x3 area
                    if (abs(i) <= 1 && abs(j) <= 1)
                        continue;

                    int2 pos = clamp(pixelPos + int2(i, j), 0, gRectSizeMinusOne);
                    float d = gIn_DiffFast[pos].x;

                    m1 += d;
                    m2 += d * d;
                }
            }

            float invNorm = 1.0 / ((REBLUR_ANTI_FIREFLY_FILTER_RADIUS * 2 + 1) * (REBLUR_ANTI_FIREFLY_FILTER_RADIUS * 2 + 1) - 3 * 3);
            m1 *= invNorm;
            m2 *= invNorm;

            float sigma = GetStdDev(m1, m2) * REBLUR_ANTI_FIREFLY_SIGMA_SCALE;
            diffLuma = clamp(diffLuma, m1 - sigma, m1 + sigma);
        } else if (gAntiFirefly && NRD_SUPPORTS_ANTIFIREFLY == 1 && diffLuma > 0.001 && !smoothTransmission) {
            float m1 = 0.0, m2 = 0.0, largest = 0.0, secondLargest = 0.0;
            [unroll] for (j = 0; j <= BORDER * 2; j++) {
                [unroll] for (i = 0; i <= BORDER * 2; i++) {
                    if (i == BORDER && j == BORDER)
                        continue;
                    float value = max(Denanify(1.0, s_DiffLuma[threadPos.y + j][threadPos.x + i]), 0.0);
                    m1 += value;
                    m2 += value * value;
                    secondLargest = max(secondLargest, min(largest, value));
                    largest = max(largest, value);
                }
            }
            const float invTrimmedCount = 1.0 / ((BORDER * 2 + 1) * (BORDER * 2 + 1) - 2);
            m1 = (m1 - largest) * invTrimmedCount;
            m2 = (m2 - largest * largest) * invTrimmedCount;
            float robustUpper = max(secondLargest * 2.0, m1 + GetStdDev(m1, m2) * 3.0) + 0.001;
            diffLuma = min(diffLuma, robustUpper);
        }

        // Fast history
        diffM1 /= diffWeight;
        diffM2 /= diffWeight;

        float diffSigma = GetStdDev(diffM1, diffM2) * gFastHistoryClampingSigmaScale;
        float diffMin = min(diffM1 - diffSigma, diffCenter);
        float diffMax = max(diffM1 + diffSigma, diffCenter);

        float diffLumaClamped = clamp(diffLuma, diffMin, diffMax);
        float diffClampingRelaxation = 1.0 / (1.0 + float(gMaxFastAccumulatedFrameNum < gMaxAccumulatedFrameNum) * frameNum.x * 2.0);
        if (gEnableHalfRateTransmission != 0 && materialID < 0.5 && roughness <= 0.12)
            diffClampingRelaxation = max(diffClampingRelaxation,
                saturate((frameNum.x - gMaxFastAccumulatedFrameNum) / max(gMaxAccumulatedFrameNum - gMaxFastAccumulatedFrameNum, 1.0)));
        diffLuma = lerp(diffLumaClamped, diffLuma, diffClampingRelaxation);

// Change luma
#    if (REBLUR_SHOW == REBLUR_SHOW_FAST_HISTORY)
        diffLuma = diffCenter;
#    endif

        diff = ChangeLuma(diff, diffLuma);
#    if (NRD_MODE == SH)
        diffSh.xyz *= GetLumaScale(length(diffSh.xyz), diffLuma);
#    endif

        // Output
        gOut_Diff[pixelPos] = diff;
#    if (NRD_MODE == SH)
        gOut_DiffSh[pixelPos] = diffSh;
#    endif
    }
#endif

// Specular
#if (NRD_SPEC)
    {
        REBLUR_TYPE spec = gIn_Spec[pixelPos];
#    if (NRD_MODE == SH)
        float4 specSh = gIn_SpecSh[pixelPos];
#    endif

        float smc = GetSpecMagicCurve(roughness);
        float specNonLinearAccumSpeed = 1.0 / (1.0 + frameNum.y);
        float robustLowHistorySpecularStrength = 0.0;
        bool transparentLowRoughness = materialID > 0.5 && roughness <= 0.45;
        bool opaqueLowRoughness = materialID < 0.5 && roughness <= 0.12;
        if (gEnableLowRoughnessSpecularStabilization != 0 && (transparentLowRoughness || opaqueLowRoughness)) {
            robustLowHistorySpecularStrength = saturate(1.0 - frameNum.y / 24.0);
        }
        bool useRobustLowHistorySpecular = robustLowHistorySpecularStrength > 0.0;

        float hitDistScale = _REBLUR_GetHitDistanceNormalization(viewZ, gHitDistParams, roughness);
        float hitDist = ExtractHitDist(spec) * hitDistScale;
#    if (NRD_MODE != OCCLUSION)
        // "gIn_SpecHitDistForTracking" is better for low roughness, but doesn't suit for high roughness ( because it's min )
        hitDist = lerp(gIn_SpecHitDistForTracking[pixelPos], hitDist, smc);
#    endif
        float hitDistFactor = GetHitDistFactor(hitDist, frustumSize);
        hitDist = saturate(hitDist / hitDistScale);

        // Stride between taps
        float specStride = stride.y;
        specStride *= lerp(0.25 + 0.75 * Math::Sqrt01(hitDistFactor), 1.0, specNonLinearAccumSpeed); // "hitDistFactor" is very noisy and breaks nice patterns
        specStride *= lerp(0.25, 1.0, smc);                                                          // hand tuned // TODO: use "lobeRadius"?
        specStride = round(specStride);

        // History reconstruction
        if (specStride != 0) {
            // Parameters
            float normalWeightParam = GetNormalWeightParam(specNonLinearAccumSpeed, gLobeAngleFraction, roughness);
            float2 geometryWeightParams = GetGeometryWeightParams(gPlaneDistSensitivity, frustumSize, Xv, Nv, specNonLinearAccumSpeed);
            float hitDistWeightNorm = 1.0 / (lerp(0.005, 0.5, smc) * specNonLinearAccumSpeed);
            float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams(roughness * roughness, sqrt(gRoughnessFraction));

            float sums = 1.0 + frameNum.y;
#    if (REBLUR_PERFORMANCE_MODE == 1)
            sums = 1.0 + 1.0 / (1.0 + gMaxAccumulatedFrameNum) - specNonLinearAccumSpeed;
#    endif

            float robustLogLumaSum = 0.0;
            float robustWeightSum = 0.0;
            if (useRobustLowHistorySpecular) {
                robustLogLumaSum = log2(max(GetLuma(spec), 1e-5)) * sums;
                robustWeightSum = sums;
            }

            spec *= sums;
#    if (NRD_MODE == SH)
            specSh.xyz *= sums;
#    endif

            [unroll] for (j = -REBLUR_HISTORY_FIX_FILTER_RADIUS; j <= REBLUR_HISTORY_FIX_FILTER_RADIUS; j++) {
                [unroll] for (i = -REBLUR_HISTORY_FIX_FILTER_RADIUS; i <= REBLUR_HISTORY_FIX_FILTER_RADIUS; i++) {
                    // Skip center
                    if (i == 0 && j == 0)
                        continue;

                    // Skip corners
                    if (abs(i) + abs(j) == REBLUR_HISTORY_FIX_FILTER_RADIUS * 2)
                        continue;

                    // Sample uv ( at the pixel center )
                    float2 uv = pixelUv + float2(i, j) * specStride * gRectSizeInv;

                    // Apply "mirror" to not waste taps going outside of the screen
                    uv = MirrorUv(uv);

                    // "uv" to "pos"
                    int2 pos = uv * gRectSize; // "uv" can't be "1"

                    // Fetch data
                    float zs = UnpackViewZ(gIn_ViewZ[WithRectOrigin(pos)]);
                    float3 Xvs = Geometry::ReconstructViewPosition(uv, gFrustum, zs, gOrthoMode);

                    float materialIDs;
                    float4 Ns = gIn_Normal_Roughness[WithRectOrigin(pos)];
                    Ns = NRD_FrontEnd_UnpackNormalAndRoughness(Ns, materialIDs);

                    // Weight
                    float angle = Math::AcosApprox(dot(Ns.xyz, N));
                    float NoX = dot(Nv, Xvs);

                    float w = ComputeWeight(NoX, geometryWeightParams.x, geometryWeightParams.y);
                    w *= CompareMaterials(materialID, materialIDs, gSpecMinMaterial);
                    w *= ComputeExponentialWeight(angle, normalWeightParam, 0.0);
                    w *= ComputeExponentialWeight(Ns.w * Ns.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y);
                    w = zs < gDenoisingRange && dot(N, Ns.xyz) >= 0.5 ? w : 0.0; // |NoX| can be ~0 if "zs" is out of range
                                                        // gaussian weight is not needed

#    if (REBLUR_PERFORMANCE_MODE == 0)
                    w *= 1.0 + UnpackData1(gIn_Data1[pos]).y;
#    endif

                    REBLUR_TYPE s = gIn_Spec[pos];
                    s = Denanify(w, s);

                    // A-trous weight
                    float hs = ExtractHitDist(s);
                    float d = hs - hitDist; // use normalized hit distances for simplicity ( no difference, roughness weight handles the rest )
                    w *= exp(-d * d * hitDistWeightNorm);

                    if (useRobustLowHistorySpecular) {
                        robustLogLumaSum += log2(max(GetLuma(s), 1e-5)) * w;
                        robustWeightSum += w;
                    }

                    // Accumulate
                    sums += w;

                    spec += s * w;
#    if (NRD_MODE == SH)
                    float4 sh = gIn_SpecSh[pos];
                    sh = Denanify(w, sh);
                    specSh.xyz += sh.xyz * w;
#    endif
                }
            }

            sums = Math::PositiveRcp(sums);
            spec *= sums;
#    if (NRD_MODE == SH)
            specSh.xyz *= sums;
#    endif

            if (useRobustLowHistorySpecular && robustWeightSum > NRD_EPS) {
                float robustLuma = exp2(robustLogLumaSum / robustWeightSum);
                float linearLuma = GetLuma(spec);
                float robustUpper = robustLuma * 1.20 + 0.0002;
                float filteredLuma = lerp(linearLuma, min(linearLuma, robustUpper),
                    robustLowHistorySpecularStrength);
                spec = ChangeLuma(spec, filteredLuma);
            }
        }

        // Local variance
        float specCenter = s_SpecLuma[threadPos.y + BORDER][threadPos.x + BORDER];
        float specM1 = specCenter;
        float specM2 = specM1 * specM1;
        float specWeight = 1.0;

        float f = frameNumAvgNorm.y;
        f = lerp(1.0, f, smc); // HistoryFix-ed data is undesired in fast history for low roughness ( test 115 )
        specCenter = lerp(GetLuma(spec), specCenter, f);
        gOut_SpecFast[pixelPos] = specCenter;

        [unroll] for (j = 0; j <= BORDER * 2; j++) {
            [unroll] for (i = 0; i <= BORDER * 2; i++) {
                // Skip center
                if (i == BORDER && j == BORDER)
                    continue;

                int2 pos = threadPos + int2(i, j);

                float s = s_SpecLuma[pos.y][pos.x];
                float w = GetSurfaceWeight(pos, int2(pixelPos) + int2(i, j) - BORDER, Xv, Nv, N, materialID, gSpecMinMaterial);
                specM1 += s * w;
                specM2 += s * s * w;
                specWeight += w;
            }
        }

        float specLuma = GetLuma(spec);

        // Anti-firefly. Rough opaque reflections use the same local robust
        // tile strategy as diffuse below; the wide linear ring is too easily
        // biased by sparse continuation hits on dark cave materials.
        bool roughOpaqueSpecular = materialID < 0.5 && roughness > 0.12;
        if (gAntiFirefly && NRD_SUPPORTS_ANTIFIREFLY == 1 && !opaqueLowRoughness && !roughOpaqueSpecular) {
            float m1 = 0;
            float m2 = 0;

            [unroll] for (j = -REBLUR_ANTI_FIREFLY_FILTER_RADIUS; j <= REBLUR_ANTI_FIREFLY_FILTER_RADIUS; j++) {
                [unroll] for (i = -REBLUR_ANTI_FIREFLY_FILTER_RADIUS; i <= REBLUR_ANTI_FIREFLY_FILTER_RADIUS; i++) {
                    // Skip central 3x3 area
                    if (abs(i) <= 1 && abs(j) <= 1)
                        continue;

                    int2 pos = clamp(pixelPos + int2(i, j), 0, gRectSizeMinusOne);
                    float s = gIn_SpecFast[pos].x;

                    if (useRobustLowHistorySpecular)
                        s = log2(max(s, 1e-5));

                    m1 += s;
                    m2 += s * s;
                }
            }

            float invNorm = 1.0 / ((REBLUR_ANTI_FIREFLY_FILTER_RADIUS * 2 + 1) * (REBLUR_ANTI_FIREFLY_FILTER_RADIUS * 2 + 1) - 3 * 3);
            m1 *= invNorm;
            m2 *= invNorm;

            float sigmaScale = useRobustLowHistorySpecular ? 0.45 : REBLUR_ANTI_FIREFLY_SIGMA_SCALE;
            float sigma = GetStdDev(m1, m2) * sigmaScale;
            if (useRobustLowHistorySpecular) {
                float ringUpper = exp2(m1 + sigma);
                float neighborMean = max((specM1 - specCenter) / max(specWeight - 1.0, 1.0), 0.0);
                float coherentUpper = neighborMean * 1.30 + 0.0002;
                float filteredLuma = min(specLuma, max(ringUpper, coherentUpper));
                specLuma = lerp(specLuma, filteredLuma, robustLowHistorySpecularStrength);
            } else
                specLuma = clamp(specLuma, m1 - sigma, m1 + sigma);
        } else if (gAntiFirefly && NRD_SUPPORTS_ANTIFIREFLY == 1 && roughOpaqueSpecular && specLuma > 0.001) {
            float m1 = 0.0, m2 = 0.0, largest = 0.0, secondLargest = 0.0;
            [unroll] for (j = 0; j <= BORDER * 2; j++) {
                [unroll] for (i = 0; i <= BORDER * 2; i++) {
                    if (i == BORDER && j == BORDER)
                        continue;
                    float value = max(Denanify(1.0, s_SpecLuma[threadPos.y + j][threadPos.x + i]), 0.0);
                    m1 += value;
                    m2 += value * value;
                    secondLargest = max(secondLargest, min(largest, value));
                    largest = max(largest, value);
                }
            }
            const float invTrimmedCount = 1.0 / ((BORDER * 2 + 1) * (BORDER * 2 + 1) - 2);
            m1 = (m1 - largest) * invTrimmedCount;
            m2 = (m2 - largest * largest) * invTrimmedCount;
            float robustUpper = max(secondLargest * 2.0, m1 + GetStdDev(m1, m2) * 3.0) + 0.001;
            specLuma = min(specLuma, robustUpper);
        }

        // Fast history
        specM1 /= specWeight;
        specM2 /= specWeight;

        float fastHistoryClampingSigmaScale = gFastHistoryClampingSigmaScale;
        if (materialID == gStrandMaterialID)
            fastHistoryClampingSigmaScale = max(fastHistoryClampingSigmaScale, 3.0);

        float specSigma = GetStdDev(specM1, specM2) * fastHistoryClampingSigmaScale;
        float specMin = min(specM1 - specSigma, specCenter);
        float specMax = max(specM1 + specSigma, specCenter);

        float specLumaClamped = clamp(specLuma, specMin, specMax);
        if (opaqueLowRoughness)
            specLumaClamped = specLuma;
        specLuma = lerp(specLumaClamped, specLuma, 1.0 / (1.0 + float(gMaxFastAccumulatedFrameNum < gMaxAccumulatedFrameNum) * frameNum.y * 2.0));

// Change luma
#    if (REBLUR_SHOW == REBLUR_SHOW_FAST_HISTORY)
        specLuma = specCenter;
#    endif

        spec = ChangeLuma(spec, specLuma);
#    if (NRD_MODE == SH)
        specSh.xyz *= GetLumaScale(length(specSh.xyz), specLuma);
#    endif

        // Output
        gOut_Spec[pixelPos] = spec;
#    if (NRD_MODE == SH)
        gOut_SpecSh[pixelPos] = specSh;
#    endif
    }
#endif
}
