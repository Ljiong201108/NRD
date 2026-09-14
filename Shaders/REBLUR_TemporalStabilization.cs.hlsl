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
#include "REBLUR_TemporalStabilization.resources.hlsli"

#include "Common.hlsli"

#include "REBLUR_Common.hlsli"

groupshared float s_DiffLuma[BUFFER_Y][BUFFER_X];
groupshared float s_SpecLuma[BUFFER_Y][BUFFER_X];
#if (NRD_DIFF && (NRD_MODE == RADIANCE || NRD_MODE == SH))
groupshared float s_DiffCurrentLuma[BUFFER_Y][BUFFER_X];
#endif
groupshared float4 s_Normal_Roughness[BUFFER_Y][BUFFER_X];
groupshared float2 s_ViewZ_Material[BUFFER_Y][BUFFER_X];

void Preload(uint2 sharedPos, int2 globalPos) {
    globalPos = clamp(globalPos, 0, gRectSizeMinusOne);

    float materialID;
    s_Normal_Roughness[sharedPos.y][sharedPos.x] = NRD_FrontEnd_UnpackNormalAndRoughness(
        gIn_Normal_Roughness[WithRectOrigin(globalPos)], materialID);
    s_ViewZ_Material[sharedPos.y][sharedPos.x] = float2(
        UnpackViewZ(gIn_ViewZ[WithRectOrigin(globalPos)]), materialID);

#if (NRD_DIFF)
    s_DiffLuma[sharedPos.y][sharedPos.x] = GetLuma(gIn_Diff[globalPos]);
#endif

#if (NRD_DIFF && (NRD_MODE == RADIANCE || NRD_MODE == SH))
    s_DiffCurrentLuma[sharedPos.y][sharedPos.x] = max(gIn_DiffCurrentLuma[globalPos], 0.0);
#endif

#if (NRD_SPEC)
    s_SpecLuma[sharedPos.y][sharedPos.x] = GetLuma(gIn_Spec[globalPos]);
#endif
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
        return; // IMPORTANT: no data output, must be rejected by the "viewZ" check!

    // Position
    float2 pixelUv = float2(pixelPos + 0.5) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition(pixelUv, gFrustum, viewZ, gOrthoMode);
    float3 X = Geometry::RotateVector(gViewToWorld, Xv);

    // Previous position and surface motion uv
    float4 inMv = gInOut_Mv[WithRectOrigin(pixelPos)];
    float3 mv = inMv.xyz * gMvScale.xyz;
    float3 Xprev = X;
    float2 smbPixelUv = pixelUv + mv.xy;

    if (gMvScale.w == 0.0) {
        if (gMvScale.z == 0.0)
            mv.z = Geometry::AffineTransform(gWorldToViewPrev, X).z - viewZ;

        float viewZprev = viewZ + mv.z;
        float3 Xvprevlocal = Geometry::ReconstructViewPosition(smbPixelUv, gFrustumPrev, viewZprev, gOrthoMode); // TODO: use gOrthoModePrev

        Xprev = Geometry::RotateVectorInverse(gWorldToViewPrev, Xvprevlocal) + gCameraDelta.xyz;
    } else {
        Xprev += mv;
        smbPixelUv = Geometry::GetScreenUv(gWorldToClipPrev, Xprev);
    }

    int2 smemPos = threadPos + BORDER;

    // Normal and roughness
    float4 normalAndRoughness = s_Normal_Roughness[smemPos.y][smemPos.x];
    float materialID = s_ViewZ_Material[smemPos.y][smemPos.x].y;
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    // Shared data
    uint bits;
    bool smbAllowCatRom;
    REBLUR_DATA1_TYPE data1 = UnpackData1(gIn_Data1[pixelPos]);
    float2 data2 = UnpackData2(gIn_Data2[pixelPos], bits, smbAllowCatRom);

    // Surface motion footprint
    float2 smbSampleUv = smbPixelUv + gHistoryJitter;
    Filtering::Bilinear smbBilinearFilter = Filtering::GetBilinearFilter(smbSampleUv, gRectSizePrev);
    float4 smbOcclusion = float4((bits & uint4(1, 2, 4, 8)) != 0);

    float4 smbOcclusionWeights = Filtering::GetBilinearCustomWeights(smbBilinearFilter, smbOcclusion);
    float smbFootprintQuality = Filtering::ApplyBilinearFilter(smbOcclusion.x, smbOcclusion.y, smbOcclusion.z, smbOcclusion.w, smbBilinearFilter);
    smbFootprintQuality = Math::Sqrt01(smbFootprintQuality);

// Diffuse
#if (NRD_DIFF)
    float diffLuma = s_DiffLuma[smemPos.y][smemPos.x];
    float diffLumaM1 = diffLuma;
    float diffLumaM2 = diffLuma * diffLuma;
    float diffGuideLumaM1 = diffLuma;
    float diffGuideLumaM2 = diffLuma * diffLuma;
    float diffGuideWeight = 1.0;
    float diffGuideLumaSpatialSupport = 1.0;
#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
    float diffCurrentLuma = s_DiffCurrentLuma[smemPos.y][smemPos.x];
    float diffCurrentGuideLumaM1 = diffCurrentLuma;
    float diffCurrentGuideLumaM2 = diffCurrentLuma * diffCurrentLuma;
    float diffCurrentGuideLumaSpatialSupport = 1.0;
#    endif

    [unroll] for (j = 0; j <= BORDER * 2; j++) {
        [unroll] for (i = 0; i <= BORDER * 2; i++) {
            if (i == BORDER && j == BORDER)
                continue;

            int2 pos = threadPos + int2(i, j);

            // Accumulate moments
            float d = s_DiffLuma[pos.y][pos.x];
            diffLumaM1 += d;
            diffLumaM2 += d * d;

            float4 sampleNormalAndRoughness = s_Normal_Roughness[pos.y][pos.x];
            float2 sampleViewZAndMaterial = s_ViewZ_Material[pos.y][pos.x];
            float relativeDepth = abs(sampleViewZAndMaterial.x - viewZ) / max(min(sampleViewZAndMaterial.x, viewZ), 0.1);
            float guideWeight = float(CompareMaterials(
                materialID, sampleViewZAndMaterial.y, gDiffMinMaterial));
            guideWeight *= Math::SmoothStep(0.88, 0.97,
                dot(sampleNormalAndRoughness.xyz, N));
            guideWeight *= 1.0 - Math::SmoothStep(0.015, 0.06, relativeDepth);
            if (materialID > 0.5)
                guideWeight *= 1.0 - Math::SmoothStep(0.08, 0.22, abs(sampleNormalAndRoughness.w - roughness));

            diffGuideLumaM1 += d * guideWeight;
            diffGuideLumaM2 += d * d * guideWeight;
            diffGuideWeight += guideWeight;
            diffGuideLumaSpatialSupport += guideWeight * float(d >= max(diffLuma * 0.35, 1e-6));

#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
            float currentD = s_DiffCurrentLuma[pos.y][pos.x];
            diffCurrentGuideLumaM1 += currentD * guideWeight;
            diffCurrentGuideLumaM2 += currentD * currentD * guideWeight;
            diffCurrentGuideLumaSpatialSupport += guideWeight * float(currentD >= max(diffCurrentLuma * 0.35, 1e-6));
#    endif
        }
    }

    // Compute sigma
    diffLumaM1 /= (BORDER * 2 + 1) * (BORDER * 2 + 1);
    diffLumaM2 /= (BORDER * 2 + 1) * (BORDER * 2 + 1);

    float diffLumaSigma = GetStdDev(diffLumaM1, diffLumaM2);
    diffGuideLumaM1 /= diffGuideWeight;
    diffGuideLumaM2 /= diffGuideWeight;
    float diffGuideLumaSigma = GetStdDev(diffGuideLumaM1, diffGuideLumaM2);
    float diffSpatialCoherence = Math::SmoothStep(0.28, 0.68,
        diffGuideLumaSpatialSupport / diffGuideWeight);
    float diffSpatialReliability = Math::SmoothStep(4.0, 10.0, diffGuideWeight);
#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
    diffCurrentGuideLumaM1 /= diffGuideWeight;
    diffCurrentGuideLumaM2 /= diffGuideWeight;
    float diffCurrentGuideLumaSigma = GetStdDev(
        diffCurrentGuideLumaM1, diffCurrentGuideLumaM2);
    float diffCurrentSpatialCoherence = Math::SmoothStep(0.28, 0.68,
        diffCurrentGuideLumaSpatialSupport / diffGuideWeight);
#    endif

    // Clean-up fireflies if HistoryFix pass was in action
    if (data1.x < gHistoryFixFrameNum)
        diffLuma = min(diffLuma, diffLumaM1 * (1.2 + 1.0 / (1.0 + data1.x)));

    // Sample history - surface motion
    float2 smbDiffHistoryState;

    BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights2(
        saturate(smbSampleUv) * gRectSizePrev, gResourceSizeInvPrev,
        smbOcclusionWeights, smbAllowCatRom,
        gHistory_DiffLumaStabilized, smbDiffHistoryState);

    // Avoid negative values
    float smbDiffLumaHistory = max(smbDiffHistoryState.x, 0.0);
    float diffRisePersistenceHistory = saturate(smbDiffHistoryState.y);
    float diffRisePersistence = 0.0;
    float currentLightingChange = 0.0;
    float diffBroadCurrentConfidence = 0.0;
    float diffLumaRiseReference = smbDiffLumaHistory;

    // Compute antilag
    // A different surface's illumination is not evidence of a lighting change.
    float diffAntilag = ComputeAntilag(smbDiffLumaHistory, diffGuideLumaM1, diffGuideLumaSigma, smbFootprintQuality * data1.x);
    float coherentLightingChange = 0.0;
    if (materialID < 0.5 && diffSpatialReliability > 0.0) {
        float coherentRise = max(
            diffGuideLumaM1 - smbDiffLumaHistory - diffGuideLumaSigma * 0.5,
            0.0);
        coherentRise /= max(
            max(diffGuideLumaM1, smbDiffLumaHistory) + diffGuideLumaSigma,
            0.002);
        coherentLightingChange = Math::SmoothStep(0.04, 0.18, coherentRise);
        coherentLightingChange *= diffSpatialCoherence * diffSpatialReliability;
        diffAntilag *= lerp(1.0, 0.20, coherentLightingChange);
    }

    // Clamp history and combine with the current frame
    float2 diffTemporalAccumulationParams = GetTemporalAccumulationParams(smbFootprintQuality, data1.x);

    float diffHistoryWeight = diffTemporalAccumulationParams.x;
    diffHistoryWeight *= diffAntilag; // this is important
    diffHistoryWeight *= float(pixelUv.x >= gSplitScreen);
    diffHistoryWeight *= float(smbPixelUv.x >= gSplitScreenPrev);

    // Include this surface's current estimate when neighboring surfaces
    // dominate the variance window (thin silhouettes and isolated pixels).
    float diffClampSigma = diffLumaSigma * diffTemporalAccumulationParams.y;
    smbDiffLumaHistory = clamp(smbDiffLumaHistory, min(diffLumaM1 - diffClampSigma, diffLuma),
        max(diffLumaM1 + diffClampSigma, diffLuma));

    float diffLumaStabilized = lerp(diffLuma, smbDiffLumaHistory, min(diffHistoryWeight, gStabilizationStrength));

    bool smoothTransmission = gEnableHalfRateTransmission != 0 && materialID > 0.5 && roughness < 0.12;
    if (gEnableLowRoughnessSpecularStabilization != 0 && !smoothTransmission) {
        float historyValidity = smbFootprintQuality * float(data1.x >= 1.0);
        historyValidity *= float(pixelUv.x >= gSplitScreen);
        historyValidity *= float(smbPixelUv.x >= gSplitScreenPrev);
        float temporalFrameScale = 2.0 / max(gFramerateScale, 1.0);
        float spatialUpper = diffGuideLumaM1 + diffGuideLumaSigma * 0.65 + 0.00015 * temporalFrameScale;

#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
        float currentCoherentRise = max(
            diffCurrentGuideLumaM1 - diffLumaRiseReference - diffCurrentGuideLumaSigma * 0.5,
            0.0);
        currentCoherentRise /= max(
            max(diffCurrentGuideLumaM1, diffLumaRiseReference) + diffCurrentGuideLumaSigma,
            0.002);
        float currentCenterAgreement = (min(diffCurrentLuma, diffCurrentGuideLumaM1) + 1e-6) / (max(diffCurrentLuma, diffCurrentGuideLumaM1) + 1e-6);
        currentLightingChange = Math::SmoothStep(
            0.04, 0.18, currentCoherentRise);
        currentLightingChange *= diffCurrentSpatialCoherence * diffSpatialReliability * Math::SmoothStep(0.25, 0.65, currentCenterAgreement);
        float diffCurrentRelativeSigma = diffCurrentGuideLumaSigma / max(diffCurrentGuideLumaM1, 0.002);
        float diffPostRelativeSigma = diffGuideLumaSigma / max(diffGuideLumaM1, 0.002);
        float diffCurrentPostAgreement = (min(diffCurrentGuideLumaM1, diffGuideLumaM1) + 1e-6) / (max(diffCurrentGuideLumaM1, diffGuideLumaM1) + 1e-6);
        diffBroadCurrentConfidence = diffCurrentSpatialCoherence * diffSpatialCoherence * diffSpatialReliability * Math::SmoothStep(0.25, 0.65, currentCenterAgreement) * (1.0 - Math::SmoothStep(0.18, 0.55, diffCurrentRelativeSigma)) * (1.0 - Math::SmoothStep(0.18, 0.55, diffPostRelativeSigma)) * Math::SmoothStep(0.20, 0.55, diffCurrentPostAgreement);
        diffBroadCurrentConfidence *= float(materialID < 0.5);
        float persistentRiseEvidence = Math::SmoothStep(
            0.20, 0.55, currentLightingChange);
        float persistenceSurfaceValidity = historyValidity * Math::SmoothStep(0.70, 0.95, smbFootprintQuality) * float(materialID < 0.5);
        if (coherentLightingChange > 0.15 && persistenceSurfaceValidity > 0.75) {
            diffRisePersistence = diffRisePersistenceHistory;
            if (persistentRiseEvidence > 0.5) {
                diffRisePersistence = min(
                    diffRisePersistence + 0.34, 1.0);
            }
        }
#    endif

        if (historyValidity <= 0.25 && diffSpatialReliability > 0.0 && diffSpatialCoherence < 0.55)
            diffLumaStabilized = min(diffLumaStabilized, spatialUpper);

        if (materialID < 0.5 && diffSpatialReliability > 0.0 && data1.x >= 2.0) {
            float historyMaturity = Math::SmoothStep(2.0, 8.0, data1.x);
            float baseSpatialResponse = lerp(0.015, 0.12, diffSpatialCoherence);
            float spatialResponse = 1.0 - pow(1.0 - baseSpatialResponse, temporalFrameScale * historyMaturity * diffSpatialReliability);
            diffLumaStabilized = lerp(
                diffLumaStabilized, diffGuideLumaM1, spatialResponse);
        }

        float motionInPixels = length(
            (smbPixelUv - pixelUv) * gRectSize);
        float staticSurfaceConfidence = historyValidity * Math::SmoothStep(0.75, 0.98, smbFootprintQuality) * (1.0 - Math::SmoothStep(0.25, 1.50, motionInPixels)) * Math::SmoothStep(4.0, 10.0, data1.x) * diffSpatialReliability * (1.0 - coherentLightingChange);
        if (materialID < 0.5 && staticSurfaceConfidence > 0.0) {
            float spatialLower = max(
                diffGuideLumaM1 - diffGuideLumaSigma * 0.65 - 0.00015 * temporalFrameScale,
                0.0);
            float historySpatialSum = clamp(
                diffLumaRiseReference, spatialLower, spatialUpper);
            float historySpatialWeight = 1.0;
            const int2 historyOffsets[4] = {
                int2(-2, 0), int2(2, 0),
                int2(0, -2), int2(0, 2)};

            [unroll] for (uint historyTap = 0; historyTap < 4; historyTap++) {
                int2 offset = historyOffsets[historyTap];
                int2 guidePos = smemPos + offset;
                int2 tapPixelPos = clamp(
                    int2(pixelPos) + offset, 0, gRectSizeMinusOne);
                float4 sampleNormalAndRoughness = s_Normal_Roughness[guidePos.y][guidePos.x];
                float2 sampleViewZAndMaterial = s_ViewZ_Material[guidePos.y][guidePos.x];
                float relativeDepth = abs(sampleViewZAndMaterial.x - viewZ) / max(min(sampleViewZAndMaterial.x, viewZ), 0.1);
                float guideWeight = float(CompareMaterials(
                    materialID, sampleViewZAndMaterial.y, gDiffMinMaterial));
                guideWeight *= Math::SmoothStep(0.88, 0.97,
                    dot(sampleNormalAndRoughness.xyz, N));
                guideWeight *= 1.0 - Math::SmoothStep(0.015, 0.06, relativeDepth);
                REBLUR_DATA1_TYPE tapData1 = UnpackData1(
                    gIn_Data1[tapPixelPos]);
                guideWeight *= Math::SmoothStep(4.0, 10.0, tapData1.x);
                if (guideWeight <= 0.0)
                    continue;

                float2 historyPixel = clamp(
                    smbSampleUv * gRectSizePrev + float2(offset),
                    0.5, gRectSizePrev - 0.5);
                float historyLuma = gHistory_DiffLumaStabilized.SampleLevel(
                                                                   gLinearClamp, historyPixel * gResourceSizeInvPrev, 0)
                                        .x;
                historyLuma = clamp(
                    max(historyLuma, 0.0), spatialLower, spatialUpper);
                historySpatialSum += historyLuma * guideWeight;
                historySpatialWeight += guideWeight;
            }

            float validTapConfidence = Math::SmoothStep(
                1.5, 3.0, historySpatialWeight - 1.0);
            float historySpatialResponse = 1.0 - pow(1.0 - 0.45, temporalFrameScale * staticSurfaceConfidence * validTapConfidence);
            diffLumaStabilized = lerp(
                diffLumaStabilized,
                historySpatialSum / historySpatialWeight,
                historySpatialResponse);
        }

        if (historyValidity > 0.25 && diffLumaStabilized > diffLumaRiseReference) {
            float spatialTarget = diffLuma;
            if (diffSpatialReliability > 0.0 && diffSpatialCoherence < 0.60)
                spatialTarget = min(spatialTarget, spatialUpper);

            float roughDiffuseResponse = Math::SmoothStep(0.20, 0.50, roughness);
            float maximumResponse = lerp(0.10, 0.25, roughDiffuseResponse);
            maximumResponse = materialID > 0.5 ? min(maximumResponse, 0.06) : maximumResponse;
            float temporalLightingConfirmation = Math::SmoothStep(
                0.006, 0.04,
                diffLumaRiseReference / max(diffGuideLumaM1 + diffGuideLumaSigma * 0.5, 0.002));
            temporalLightingConfirmation *= historyValidity * Math::SmoothStep(2.0, 6.0, data1.x);
            float coherentMaximumResponse = lerp(
                0.003, 0.08, temporalLightingConfirmation);
            maximumResponse = lerp(
                maximumResponse, coherentMaximumResponse,
                coherentLightingChange);
#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
            float persistentLightingConfirmation = Math::SmoothStep(
                0.80, 0.99, diffRisePersistence);
            float suspiciousLightingRise = Math::SmoothStep(
                0.35, 0.70,
                max(coherentLightingChange,
                    currentLightingChange));
            float unconfirmedLightingRise = suspiciousLightingRise * (1.0 - persistentLightingConfirmation);
            float persistentResponseGate = 1.0 - unconfirmedLightingRise;
            float opaqueResponseFloor = 0.003;
#    else
            float persistentLightingConfirmation = 1.0;
            float unconfirmedLightingRise = 0.0;
            float opaqueResponseFloor = 0.003;
#    endif
            if (materialID < 0.5) {
#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
                float opaqueResponseCap = lerp(
                    opaqueResponseFloor, 0.08,
                    coherentLightingChange * persistentResponseGate);
#    else
                float opaqueResponseCap = lerp(
                    0.003, 0.08, coherentLightingChange);
#    endif
                maximumResponse = min(
                    maximumResponse, opaqueResponseCap);
            }
            float minimumResponse = 0.003;
#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
            minimumResponse = materialID < 0.5 ? opaqueResponseFloor : minimumResponse;
#    endif
            float baseResponse = lerp(minimumResponse, maximumResponse,
                diffSpatialCoherence * diffSpatialReliability);
            float frameResponse = 1.0 - pow(1.0 - baseResponse, temporalFrameScale);
            float temporalRise = max(spatialTarget - diffLumaRiseReference, 0.0) * frameResponse;
            float absoluteAllowance = lerp(
                0.000008, 0.000015, roughDiffuseResponse);
#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
            if (materialID < 0.5)
                absoluteAllowance *= 1.0 - unconfirmedLightingRise;
#    endif
            float temporalUpper = diffLumaRiseReference + max(temporalRise, absoluteAllowance * temporalFrameScale);
#    if (NRD_MODE == RADIANCE || NRD_MODE == SH)
            float broadLightingFrameResponse = 1.0 - pow(1.0 - 0.10 * diffBroadCurrentConfidence, temporalFrameScale);
            float broadLightingUpper = diffLumaRiseReference + max(diffLumaStabilized - diffLumaRiseReference, 0.0) * broadLightingFrameResponse;
            temporalUpper = max(temporalUpper, broadLightingUpper);
#    endif
            diffLumaStabilized = min(diffLumaStabilized, temporalUpper);
        }

        if (diffSpatialReliability > 0.0 && diffLumaRiseReference > spatialUpper)
            diffLumaStabilized = min(diffLumaStabilized, max(diffLuma, spatialUpper));
    }

    REBLUR_TYPE diff = gIn_Diff[pixelPos];
    diff = ChangeLuma(diff, diffLumaStabilized);
#    if (NRD_MODE == SH)
    REBLUR_SH_TYPE diffSh = gIn_DiffSh[pixelPos];
    diffSh.xyz *= GetLumaScale(length(diffSh.xyz), diffLumaStabilized);
#    endif

    // Output
    diff.w = gReturnHistoryLengthInsteadOfOcclusion ? data1.x : diff.w;

    gOut_Diff[pixelPos] = diff;
    gOut_DiffLumaStabilized[pixelPos] = float2(
        diffLumaStabilized, diffRisePersistence);
#    if (NRD_MODE == SH)
    gOut_DiffSh[pixelPos] = diffSh;
#    endif

    // Increment history length
    data1.x += 1.0;

    // Apply anti-lag
    float diffMinAccumSpeed = min(data1.x, gHistoryFixFrameNum) * REBLUR_USE_ANTILAG_NOT_INVOKING_HISTORY_FIX;
    data1.x = lerp(diffMinAccumSpeed, data1.x, diffAntilag);
#endif

// Specular
#if (NRD_SPEC)
    float specLuma = s_SpecLuma[smemPos.y][smemPos.x];
    float specLumaM1 = specLuma;
    float specLumaM2 = specLuma * specLuma;
    float specLumaSpatialSupport = 1.0;
    float specGuideLumaM1 = specLuma;
    float specGuideLumaM2 = specLuma * specLuma;
    float specGuideWeight = 1.0;
    float specGuideLumaSpatialSupport = 1.0;

    [unroll] for (j = 0; j <= BORDER * 2; j++) {
        [unroll] for (i = 0; i <= BORDER * 2; i++) {
            if (i == BORDER && j == BORDER)
                continue;

            int2 pos = threadPos + int2(i, j);

            // Accumulate moments
            float s = s_SpecLuma[pos.y][pos.x];
            specLumaM1 += s;
            specLumaM2 += s * s;
            specLumaSpatialSupport += float(
                s >= max(specLuma * 0.35, 1e-6));

            float4 sampleNormalAndRoughness = s_Normal_Roughness[pos.y][pos.x];
            float2 sampleViewZAndMaterial = s_ViewZ_Material[pos.y][pos.x];
            float relativeDepth = abs(sampleViewZAndMaterial.x - viewZ) / max(min(sampleViewZAndMaterial.x, viewZ), 0.1);
            float guideWeight = float(CompareMaterials(
                materialID, sampleViewZAndMaterial.y, gSpecMinMaterial));
            guideWeight *= Math::SmoothStep(0.88, 0.97,
                dot(sampleNormalAndRoughness.xyz, N));
            guideWeight *= 1.0 - Math::SmoothStep(0.015, 0.06, relativeDepth);
            guideWeight *= 1.0 - Math::SmoothStep(0.025, 0.10, abs(sampleNormalAndRoughness.w - roughness));

            specGuideLumaM1 += s * guideWeight;
            specGuideLumaM2 += s * s * guideWeight;
            specGuideWeight += guideWeight;
            specGuideLumaSpatialSupport += guideWeight * float(s >= max(specLuma * 0.35, 1e-6));
        }
    }

    // Compute sigma
    specLumaM1 /= (BORDER * 2 + 1) * (BORDER * 2 + 1);
    specLumaM2 /= (BORDER * 2 + 1) * (BORDER * 2 + 1);

    float specLumaSigma = GetStdDev(specLumaM1, specLumaM2);
    specGuideLumaM1 /= specGuideWeight;
    specGuideLumaM2 /= specGuideWeight;
    float specGuideLumaSigma = GetStdDev(
        specGuideLumaM1, specGuideLumaM2);
    float specUnguidedSpatialCoherence = Math::SmoothStep(0.24, 0.64,
        specLumaSpatialSupport / float((BORDER * 2 + 1) * (BORDER * 2 + 1)));
    float specSpatialCoherence = Math::SmoothStep(0.24, 0.64,
        specGuideLumaSpatialSupport / specGuideWeight);
    float specSpatialReliability = Math::SmoothStep(4.0, 10.0, specGuideWeight);
    float specGuideRelativeSigma = specGuideLumaSigma / max(specGuideLumaM1, 5e-5);
    float specGuideCenterAgreement = (min(specLuma, specGuideLumaM1) + 5e-5) / (max(specLuma, specGuideLumaM1) + 5e-5);
    float specBroadChangeConfidence = specSpatialCoherence * specSpatialReliability * (1.0 - Math::SmoothStep(0.55, 1.20, specGuideRelativeSigma)) * Math::SmoothStep(0.30, 0.70, specGuideCenterAgreement);

    // Clean-up fireflies if HistoryFix pass was in action
    if (data1.y < gHistoryFixFrameNum)
        specLuma = min(specLuma, specLumaM1 * (1.2 + 1.0 / (1.0 + data1.y)));

    // Hit distance for tracking ( tests 6, 67, 155 )
    REBLUR_TYPE spec = gIn_Spec[pixelPos];
    float hitDistForTracking = ExtractHitDist(spec) * _REBLUR_GetHitDistanceNormalization(viewZ, gHitDistParams, roughness); // TODO: min in 3x3 seems to be not needed here

    // Needed to preserve contact ( test 3, 8 ), but adds pixelation in some cases ( test 160 ). More fun if lobe trimming is off.
    [flatten] if (gSpecPrepassBlurRadius != 0.0)
        hitDistForTracking = min(hitDistForTracking, gIn_SpecHitDistForTracking[pixelPos]);

    // Virtual motion
    float virtualHistoryAmount = data2.x;
    float curvature = data2.y;

    float3 V = GetViewVector(X);
    float NoV = abs(dot(N, V));
    float3 Xvirtual = GetXvirtual(hitDistForTracking, curvature, X, Xprev, N, V, roughness);

    float2 vmbPixelUv = Geometry::GetScreenUv(gWorldToClipPrev, Xvirtual);
    vmbPixelUv = materialID == gCameraAttachedReflectionMaterialID ? pixelUv : vmbPixelUv;

    // Modify MVs if requested
    if (gSpecProbabilityThresholdsForMvModification.x < 1.0 && NRD_SUPPORTS_BASECOLOR_METALNESS) {
        float4 baseColorMetalness = gIn_BaseColor_Metalness[WithRectOrigin(pixelPos)];

        float3 albedo, Rf0;
        BRDF::ConvertBaseColorMetalnessToAlbedoRf0(baseColorMetalness.xyz, baseColorMetalness.w, albedo, Rf0);

        float3 Fenv = BRDF::EnvironmentTerm_Rtg(Rf0, NoV, roughness);

        float lumSpec = Color::Luminance(Fenv);
        float lumDiff = Color::Luminance(albedo * (1.0 - Fenv));
        float specProb = lumSpec / (lumDiff + lumSpec + NRD_EPS);

        float f = Math::SmoothStep(gSpecProbabilityThresholdsForMvModification.x, gSpecProbabilityThresholdsForMvModification.y, specProb);
        f *= 1.0 - GetSpecMagicCurve(roughness);
        f *= 1.0 - Math::Sqrt01(abs(curvature));

        if (f != 0.0) {
            float3 specMv = Xvirtual - X; // world-space delta fits badly into FP16! Prefer 2.5D motion!
            if (gMvScale.w == 0.0) {
                specMv.xy = vmbPixelUv - pixelUv;
                specMv.z = Geometry::AffineTransform(gWorldToViewPrev, Xvirtual).z - viewZ; // TODO: is it useful?
            }

            // Modify only .xy for 2D and .xyz for 2.5D and 3D MVs
            mv.xy = specMv.xy / gMvScale.xy;
            mv.z = gMvScale.z == 0.0 ? inMv.z : specMv.z / gMvScale.z;

            inMv.xyz = lerp(inMv.xyz, mv, f);

            gInOut_Mv[WithRectOrigin(pixelPos)] = inMv;
        }
    }

    // Sample history - surface motion
    float smbSpecLumaHistory;

    BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights1(
        saturate(smbSampleUv) * gRectSizePrev, gResourceSizeInvPrev,
        smbOcclusionWeights, smbAllowCatRom,
        gHistory_SpecLumaStabilized, smbSpecLumaHistory);

    // Virtual motion footprint
    float2 vmbSampleUv = vmbPixelUv + gHistoryJitter;
    Filtering::Bilinear vmbBilinearFilter = Filtering::GetBilinearFilter(vmbSampleUv, gRectSizePrev);
    float4 vmbOcclusion = float4((bits & uint4(16, 32, 64, 128)) != 0);
    float4 vmbOcclusionWeights = Filtering::GetBilinearCustomWeights(vmbBilinearFilter, vmbOcclusion);
    bool vmbAllowCatRom = dot(vmbOcclusion, 1.0) > 3.5 && REBLUR_USE_CATROM_FOR_VIRTUAL_MOTION_IN_TS;
    vmbAllowCatRom = vmbAllowCatRom && smbAllowCatRom; // helps to reduce over-sharpening in disoccluded areas
    float vmbFootprintQuality = Filtering::ApplyBilinearFilter(vmbOcclusion.x, vmbOcclusion.y, vmbOcclusion.z, vmbOcclusion.w, vmbBilinearFilter);
    vmbFootprintQuality = Math::Sqrt01(vmbFootprintQuality);

    // Sample history - virtual motion
    float vmbSpecLumaHistory;

    BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights1(
        saturate(vmbSampleUv) * gRectSizePrev, gResourceSizeInvPrev,
        vmbOcclusionWeights, vmbAllowCatRom,
        gHistory_SpecLumaStabilized, vmbSpecLumaHistory);

    // Avoid negative values
    smbSpecLumaHistory = max(smbSpecLumaHistory, 0.0);
    vmbSpecLumaHistory = max(vmbSpecLumaHistory, 0.0);

    // Combine surface and virtual motion
    float specLumaHistory = lerp(smbSpecLumaHistory, vmbSpecLumaHistory, virtualHistoryAmount);
    float specLumaRiseReference = specLumaHistory;
    float footprintQuality = lerp(smbFootprintQuality, vmbFootprintQuality, virtualHistoryAmount);

    // Compute antilag
    float specAntilag = ComputeAntilag(specLumaHistory, specGuideLumaM1, specGuideLumaSigma, footprintQuality * data1.y);

    // Clamp history and combine with the current frame
    float2 specTemporalAccumulationParams = GetTemporalAccumulationParams(footprintQuality, data1.y);

    // TODO: roughness should affect stabilization:
    // - use "virtualHistoryRoughnessBasedConfidence" from TA
    // - compute moments for samples with similar roughness
    float specHistoryWeight = specTemporalAccumulationParams.x;
    specHistoryWeight *= specAntilag; // this is important
    specHistoryWeight *= float(pixelUv.x >= gSplitScreen);
    specHistoryWeight *= virtualHistoryAmount != 1.0 ? float(smbPixelUv.x >= gSplitScreenPrev) : 1.0;
    specHistoryWeight *= virtualHistoryAmount != 0.0 ? float(vmbPixelUv.x >= gSplitScreenPrev) : 1.0;

    float responsiveFactor = RemapRoughnessToResponsiveFactor(roughness);
    float smc = GetSpecMagicCurve(roughness);
    float acceleration = lerp(smc, 1.0, 0.5 + responsiveFactor * 0.5);
    if (materialID == gStrandMaterialID)
        acceleration = min(acceleration, 0.5);

    specHistoryWeight *= acceleration;

    // Include this surface's current estimate when neighboring surfaces
    // dominate the variance window (thin silhouettes and isolated pixels).
    float specClampSigma = specLumaSigma * specTemporalAccumulationParams.y;
    specLumaHistory = clamp(specLumaHistory, min(specLumaM1 - specClampSigma, specLuma),
        max(specLumaM1 + specClampSigma, specLuma));

    float specLumaStabilized = lerp(specLuma, specLumaHistory, min(specHistoryWeight, gStabilizationStrength));

    bool transparentGlossy = materialID > 0.5 && materialID < 2.5 && roughness <= 0.45;
    bool opaqueLowRoughness = materialID < 0.5 && roughness <= 0.25;
    if (gEnableLowRoughnessSpecularStabilization != 0 && (transparentGlossy || opaqueLowRoughness)) {
        float historyValidity = footprintQuality * float(data1.y >= 1.0);
        historyValidity *= float(pixelUv.x >= gSplitScreen);
        historyValidity *= virtualHistoryAmount != 1.0 ? float(smbPixelUv.x >= gSplitScreenPrev) : 1.0;
        historyValidity *= virtualHistoryAmount != 0.0 ? float(vmbPixelUv.x >= gSplitScreenPrev) : 1.0;
        float temporalFrameScale = 2.0 / max(gFramerateScale, 1.0);
        float customSpatialCoherence = transparentGlossy ? specUnguidedSpatialCoherence : specSpatialCoherence;
        float customSpatialReliability = transparentGlossy ? 1.0 : specSpatialReliability;
        float spatialUpper = transparentGlossy ? specLumaM1 + specLumaSigma * 0.50 + 0.00008 * temporalFrameScale : specGuideLumaM1 + specGuideLumaSigma * 0.50 + 0.00025 * temporalFrameScale;

        if (historyValidity <= 0.25 && customSpatialReliability > 0.0) {
            float disocclusionUpper = transparentGlossy ? spatialUpper : specGuideLumaM1 + specGuideLumaSigma * 0.30 + 0.00002 * temporalFrameScale;
            if (!opaqueLowRoughness || customSpatialCoherence < 0.55)
                specLumaStabilized = min(
                    specLumaStabilized, disocclusionUpper);
            else
                specLumaStabilized = min(
                    specLumaStabilized,
                    lerp(disocclusionUpper, spatialUpper, 0.20));
        }

        float specTemporalConfirmation = transparentGlossy ? 1.0 : Math::SmoothStep(0.15, 0.55, specLumaRiseReference / max(specGuideLumaM1 + specGuideLumaSigma * 0.5, 0.001));
        if (opaqueLowRoughness)
            specTemporalConfirmation *= historyValidity * Math::SmoothStep(2.0, 8.0, data1.y);

        if (opaqueLowRoughness && historyValidity > 0.25 && specSpatialReliability > 0.0 && specGuideLumaM1 > specLumaStabilized) {
            float darkIsland = Math::SmoothStep(
                0.04, 0.20,
                (specGuideLumaM1 - specLumaStabilized) / max(specGuideLumaM1 + specGuideLumaSigma, 0.001));
            float historyMaturity = Math::SmoothStep(2.0, 8.0, data1.y);
            float baseSpatialResponse = 0.08 * specSpatialCoherence * specSpatialReliability * darkIsland * specTemporalConfirmation;
            float spatialResponse = 1.0 - pow(1.0 - baseSpatialResponse, temporalFrameScale * historyMaturity);
            specLumaStabilized = lerp(
                specLumaStabilized, specGuideLumaM1, spatialResponse);
        }

        if (historyValidity > 0.25 && specLumaStabilized > specLumaRiseReference) {
            float spatialTarget = transparentGlossy ? specLumaStabilized : specLuma;
            if (!transparentGlossy && specSpatialReliability > 0.0 && specSpatialCoherence < 0.60)
                spatialTarget = min(spatialTarget, spatialUpper);
            float confirmedResponse = lerp(
                0.001, 0.05, specTemporalConfirmation);
            float maximumResponse = transparentGlossy ? 0.05 : lerp(confirmedResponse, 0.32, specBroadChangeConfidence);
            float minimumResponse = transparentGlossy ? 0.003 : 0.001;
            float baseResponse = lerp(minimumResponse, maximumResponse,
                customSpatialCoherence * customSpatialReliability);
            float frameResponse = 1.0 - pow(1.0 - baseResponse, temporalFrameScale);
            float temporalRise = max(
                                     spatialTarget - specLumaRiseReference, 0.0)
                * frameResponse;
            float absoluteAllowance = transparentGlossy ? 0.00003 : 0.000008;
            float temporalUpper = specLumaRiseReference + max(temporalRise, absoluteAllowance * temporalFrameScale);
            specLumaStabilized = min(specLumaStabilized, temporalUpper);
        }

        if (customSpatialReliability > 0.0 && specLumaRiseReference > spatialUpper)
            specLumaStabilized = min(specLumaStabilized, max(specLuma, spatialUpper));
    }

    spec = ChangeLuma(spec, specLumaStabilized);
#    if (NRD_MODE == SH)
    REBLUR_SH_TYPE specSh = gIn_SpecSh[pixelPos];
    specSh.xyz *= GetLumaScale(length(specSh.xyz), specLumaStabilized);
#    endif

    // Output
    spec.w = gReturnHistoryLengthInsteadOfOcclusion ? data1.y : spec.w;

    gOut_Spec[pixelPos] = spec;
    gOut_SpecLumaStabilized[pixelPos] = specLumaStabilized;
#    if (NRD_MODE == SH)
    gOut_SpecSh[pixelPos] = specSh;
#    endif

    // Increment history length
    data1.y += 1.0;

    // Apply anti-lag
    float specMinAccumSpeed = min(data1.y, gHistoryFixFrameNum) * REBLUR_USE_ANTILAG_NOT_INVOKING_HISTORY_FIX;
    data1.y = lerp(specMinAccumSpeed, data1.y, specAntilag);
#endif

    gOut_InternalData[pixelPos] = PackInternalData(data1.x, data1.y, materialID);
}
