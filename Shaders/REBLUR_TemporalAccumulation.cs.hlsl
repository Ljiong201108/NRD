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
#include "REBLUR_TemporalAccumulation.resources.hlsli"

#include "Common.hlsli"

#include "REBLUR_Common.hlsli"

groupshared float4 s_Normal_HitDistForTracking[ BUFFER_Y ][ BUFFER_X ];

float2 StochasticBilinear( float2 uv, float2 texSize )
{
    #if( REBLUR_USE_STF == 1 && NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        // Requires: Rng::Hash::Initialize( pixelPos, gFrameIndex )
        Filtering::Bilinear f = Filtering::GetBilinearFilter( uv, texSize );

        float2 rnd = Rng::Hash::GetFloat2( );
        f.origin += step( rnd, f.weights );

        return ( f.origin + 0.5 ) / texSize;
    #else
        return uv;
    #endif
}

float GetLowRoughnessSpatialWeight( int2 pos, float materialID, float3 N, float roughness, float viewZ )
{
    float zs = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pos ) ] );
    float materialIDs;
    float4 Ns = NRD_FrontEnd_UnpackNormalAndRoughness(
        gIn_Normal_Roughness[ WithRectOrigin( pos ) ], materialIDs );
    float relativeDepth = abs( zs - viewZ ) / max( min( zs, viewZ ), 0.1 );
    float w = CompareMaterials( materialID, materialIDs, gSpecMinMaterial );
    w *= Math::SmoothStep( 0.95, 0.995, dot( Ns.xyz, N ) );
    w *= 1.0 - Math::SmoothStep( 0.02, 0.08, abs( Ns.w - roughness ) );
    w *= 1.0 - Math::SmoothStep( 0.005, 0.03, relativeDepth );

    return w * float( zs < gDenoisingRange );
}

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSizeMinusOne );

    float3 N = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( globalPos ) ] ).xyz;
    float hitDistForTracking = 0.0;

    #if( NRD_SPEC )
        #if( NRD_MODE == OCCLUSION )
            uint shift = gSpecCheckerboard != 2 ? 1 : 0;
            uint2 pos = uint2( globalPos.x >> shift, globalPos.y );
        #else
            uint2 pos = globalPos;
        #endif

        REBLUR_TYPE spec = gIn_Spec[ pos ];
        #if( NRD_MODE == OCCLUSION )
            float hitDist = ExtractHitDist( spec );
        #else
            float hitDist = gSpecPrepassBlurRadius == 0.0 ? ExtractHitDist( spec ) : gIn_SpecHitDistForTracking[ globalPos ];
        #endif

        float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( globalPos ) ] );

        hitDistForTracking = ( hitDist == 0.0 || !( viewZ < gDenoisingRange ) ) ? NRD_INF : hitDist;
    #endif

    s_Normal_HitDistForTracking[ sharedPos.y ][ sharedPos.x ] = float4( N, hitDistForTracking );
}

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_DEFAULT;

    // Preload
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // Tile-based early out
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // Early out
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pixelPos ) ] );
    if( !( viewZ < gDenoisingRange ) )
        return;

    // Current position
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );
    float3 X = Geometry::RotateVector( gViewToWorld, Xv );

    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ], materialID );
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;
    // Find hit distance for tracking, averaged normal and roughness variance
    float3 Navg = 0.0; // needs to be unnormalized!
    #if( NRD_SPEC )
        float hitDistForTracking = NRD_INF;
    #endif

    [unroll]
    for( j = 0; j <= BORDER * 2; j++ )
    {
        [unroll]
        for( i = 0; i <= BORDER * 2; i++ )
        {
            int2 pos = threadPos + int2( i, j );
            float4 data = s_Normal_HitDistForTracking[ pos.y ][ pos.x ];

            // Average normal
            if( i < 2 && j < 2 )
                Navg += data.xyz * 0.25;

            #if( NRD_SPEC )
                // Min hit distance for tracking, ignoring 0 values ( which still can be produced by VNDF sampling )
                hitDistForTracking = min( hitDistForTracking, data.w );
            #endif
        }
    }

    // Normal and roughness

    bool lowRoughnessSurfaceGuide = materialID < 0.5 || materialID > 1.5;
    bool opaqueLowRoughness = materialID < 0.5 && roughness <= 0.12;
    float guidedSpecularRoughnessLimit = materialID > 1.5 ? 0.45 : 0.12;

    #if( NRD_SPEC )
        // Modified roughness is essential for "smb" specular motion
        float roughnessModified = Filtering::GetModifiedRoughnessFromNormalVariance( roughness, Navg );

        // Hit distance for tracking ( tests 8, 110, 139, e3, e9 without normal map, e24 )
        #if( REBLUR_USE_STF == 1 && NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
            Rng::Hash::Initialize( pixelPos, gFrameIndex );
        #endif

        hitDistForTracking = hitDistForTracking == NRD_INF ? 0.0 : hitDistForTracking;

        float hitDistNormalization = _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistParams, roughness );
        #if( NRD_MODE == OCCLUSION )
            hitDistForTracking *= hitDistNormalization;
        #else
            hitDistForTracking *= gSpecPrepassBlurRadius == 0.0 ? hitDistNormalization : 1.0;
        #endif

        gOut_SpecHitDistForTracking[ pixelPos ] = hitDistForTracking;
    #endif

    // Previous position and surface motion uv
    float3 mv = gIn_Mv[ WithRectOrigin( pixelPos ) ] * gMvScale.xyz;
    float3 Xprev = X;
    float2 smbPixelUv = pixelUv + mv.xy;

    if( gMvScale.w == 0.0 )
    {
        if( gMvScale.z == 0.0 )
            mv.z = Geometry::AffineTransform( gWorldToViewPrev, X ).z - viewZ;

        float viewZprev = viewZ + mv.z;
        float3 Xvprevlocal = Geometry::ReconstructViewPosition( smbPixelUv, gFrustumPrev, viewZprev, gOrthoMode ); // TODO: use gOrthoModePrev

        Xprev = Geometry::RotateVectorInverse( gWorldToViewPrev, Xvprevlocal ) + gCameraDelta.xyz;
    }
    else
    {
        Xprev += mv;
        smbPixelUv = Geometry::GetScreenUv( gWorldToClipPrev, Xprev );
    }

    float2 smbSampleUv = smbPixelUv + gHistoryJitter;
    Filtering::Bilinear smbBilinearFilter = Filtering::GetBilinearFilter( smbSampleUv, gRectSizePrev );
    float2 smbBilinearGatherUv = ( smbBilinearFilter.origin + 1.0 ) * gResourceSizeInvPrev;
    float4 prevViewZ = UnpackViewZ( gPrev_ViewZ.GatherRed( gNearestClamp, smbBilinearGatherUv ).wzxy );
    uint4 smbInternalData = gPrev_InternalData.GatherRed( gNearestClamp, smbBilinearGatherUv ).wzxy;

    float smbNoN;
    float4 smbNoN2x2;
    {
        float3 Nt = N;

        #if( NRD_USE_PREV_WORLD_SPACE_MATRIX == 1 )
            Nt = Geometry::RotateVectorInverse( gWorldPrevToWorld, Nt ); // to "prev" world space
        #endif

        int3 p = int3( smbBilinearFilter.origin, 0 );
        float3 n00 = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.Load( p ) ).xyz;
        float3 n10 = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.Load( p, int2( 1, 0 ) ) ).xyz;
        float3 n01 = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.Load( p, int2( 0, 1 ) ) ).xyz;
        float3 n11 = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.Load( p, int2( 1, 1 ) ) ).xyz;

        smbNoN2x2.x = dot( n00, Nt );
        smbNoN2x2.y = dot( n10, Nt );
        smbNoN2x2.z = dot( n01, Nt );
        smbNoN2x2.w = dot( n11, Nt );
    }

    // Parallax
    float smbParallaxInPixels1 = ComputeParallaxInPixels( Xprev + gCameraDelta.xyz, gOrthoMode == 0.0 ? smbPixelUv : pixelUv, gWorldToClipPrev, gRectSize );
    float smbParallaxInPixels2 = ComputeParallaxInPixels( Xprev - gCameraDelta.xyz, gOrthoMode == 0.0 ? pixelUv : smbPixelUv, gWorldToClip, gRectSize );

    float smbParallaxInPixelsMax = max( smbParallaxInPixels1, smbParallaxInPixels2 );
    float smbParallaxInPixelsMin = min( smbParallaxInPixels1, smbParallaxInPixels2 );

    // Disocclusion: threshold
    float pixelSize = PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, viewZ );
    float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );

    float disocclusionThresholdMix = 0;
    if( materialID == gStrandMaterialID )
        disocclusionThresholdMix = NRD_GetNormalizedStrandThickness( gStrandThickness, pixelSize );
    if( gHasDisocclusionThresholdMix && NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX )
        disocclusionThresholdMix = gIn_DisocclusionThresholdMix[ WithRectOrigin( pixelPos ) ];

    float disocclusionThreshold = lerp( gDisocclusionThreshold, gDisocclusionThresholdAlternate, disocclusionThresholdMix );
    if( materialID == gStrandMaterialID )
    {
        // Further relax "disocclusionThreshold" if parallax is relatively small
        float mediumParallax = Math::SmoothStep01( smbParallaxInPixelsMax );
        disocclusionThreshold = lerp( NRD_STRAND_RELAXED_DISOCCLUSION_THRESHOLD, disocclusionThreshold, mediumParallax );
    }

    // TODO: small parallax ( very slow motion ) could be used to increase disocclusion threshold, but:
    // - MVs should be dilated first
    // - IMPORTANT: a static pixel ( with relaxed threshold ) can touch a moving pixel, leading to reprojection artefacts
    float smallParallax = Math::LinearStep( 0.25, 0.0, smbParallaxInPixelsMax );
    float cosMaxAngle = max( REBLUR_ALMOST_ZERO_ANGLE - 0.25 * smallParallax, 0.5 );
    #if( NRD_DIFF && !NRD_SPEC && ( NRD_MODE == RADIANCE || NRD_MODE == SH ) )
        if( gEnableHalfRateTransmission != 0 && materialID > 0.5 && materialID < 1.5 && roughness < 0.12 )
            cosMaxAngle = min( cosMaxAngle, 0.9 );
    #endif

    float3 V = GetViewVector( X );
    float NoV = abs( dot( N, V ) );
    float NoVstrict = lerp( NoV, 1.0, saturate( smbParallaxInPixelsMax / 30.0 ) );

    // Disocclusion
    float4 smbDisocclusionThreshold = float4( smbNoN2x2 > cosMaxAngle ); // normal
    smbDisocclusionThreshold *= IsInScreenBilinear( smbBilinearFilter.origin, gRectSizePrev ); // in screen
    smbDisocclusionThreshold *= GetDisocclusionThreshold( disocclusionThreshold, frustumSize, NoVstrict );
    smbDisocclusionThreshold -= NRD_EPS;

    float3 Xvprev = Geometry::AffineTransform( gWorldToViewPrev, Xprev );
    float4 smbPlaneDist = abs( prevViewZ - Xvprev.z );
    float4 smbOcclusion = step( smbPlaneDist, smbDisocclusionThreshold ) * ( prevViewZ < gDenoisingRange );

    #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        float4 smbMaterialID = float4( UnpackInternalData( smbInternalData.x ).z,
            UnpackInternalData( smbInternalData.y ).z, UnpackInternalData( smbInternalData.z ).z,
            UnpackInternalData( smbInternalData.w ).z );
        smbOcclusion *= CompareMaterials( materialID, smbMaterialID, min( gSpecMinMaterial, gDiffMinMaterial ) );
    #endif

    float3 normalViewPrev = Geometry::RotateVector(gWorldToViewPrev, N);
    float4 planeTaps;
    [unroll]
    for (uint tap = 0; tap < 4; ++tap) {
        float2 tapUv = (smbBilinearFilter.origin + int2(tap & 1, tap >> 1) + 0.5) / gRectSizePrev - gHistoryJitter;
        float3 tapXv = Geometry::ReconstructViewPosition(tapUv, gFrustumPrev, prevViewZ[tap], gOrthoMode);
        planeTaps[tap] = float(abs(dot(normalViewPrev, tapXv - Xvprev)) <= max(NRD_DISOCCLUSION_THRESHOLD * viewZ * NoV, NRD_EPS));
    }
    smbOcclusion *= planeTaps;

    // 2x2 occlusion weights
    float4 smbOcclusionWeights = Filtering::GetBilinearCustomWeights( smbBilinearFilter, smbOcclusion );
    smbNoN = Filtering::ApplyBilinearCustomWeights( smbNoN2x2.x, smbNoN2x2.y, smbNoN2x2.z, smbNoN2x2.w, smbOcclusionWeights );
    bool smbAllowCatRom = false;

    float fbits = smbOcclusion.x * 1.0;
    fbits += smbOcclusion.y * 2.0;
    fbits += smbOcclusion.z * 4.0;
    fbits += smbOcclusion.w * 8.0;

    // Accumulation speed
    float2 internalData00 = UnpackInternalData( smbInternalData.x ).xy;
    float2 internalData10 = UnpackInternalData( smbInternalData.y ).xy;
    float2 internalData01 = UnpackInternalData( smbInternalData.z ).xy;
    float2 internalData11 = UnpackInternalData( smbInternalData.w ).xy;

    #if( NRD_DIFF )
        float4 diffAccumSpeeds = float4( internalData00.x, internalData10.x, internalData01.x, internalData11.x );
        float diffAccumSpeed = Filtering::ApplyBilinearCustomWeights( diffAccumSpeeds.x, diffAccumSpeeds.y, diffAccumSpeeds.z, diffAccumSpeeds.w, smbOcclusionWeights );
    #endif

    #if( NRD_SPEC )
        float4 specAccumSpeeds = float4( internalData00.y, internalData10.y, internalData01.y, internalData11.y );
        float smbSpecAccumSpeed = Filtering::ApplyBilinearCustomWeights( specAccumSpeeds.x, specAccumSpeeds.y, specAccumSpeeds.z, specAccumSpeeds.w, smbOcclusionWeights );
    #endif

    // Footprint quality
    float3 smbVprev = GetViewVectorPrev( Xprev, gCameraDelta.xyz );
    float NoVprev = abs( dot( N, smbVprev ) ); // TODO: should be "smbN", but jittering breaks logic
    float sizeQuality = ( NoVprev + 1e-3 ) / ( NoV + 1e-3 ); // this order because we need to fix stretching only, shrinking is OK
    sizeQuality *= sizeQuality;
    sizeQuality = lerp( 0.1, 1.0, saturate( sizeQuality ) );

    float smbFootprintQuality = Filtering::ApplyBilinearFilter( smbOcclusion.x, smbOcclusion.y, smbOcclusion.z, smbOcclusion.w, smbBilinearFilter );
    smbFootprintQuality = Math::Sqrt01( smbFootprintQuality );
    smbFootprintQuality *= sizeQuality; // avoid footprint momentary stretching due to changed viewing angle

    #if( NRD_SPEC )
        if( gEnableLowRoughnessSpecularStabilization != 0 && lowRoughnessSurfaceGuide && roughness <= guidedSpecularRoughnessLimit && smbFootprintQuality > 0.25 )
        {
            float previousHitDistForTracking = gPrev_SpecHitDistForTracking.SampleLevel( gLinearClamp, smbSampleUv * gResolutionScalePrev, 0 );
            if( hitDistForTracking > NRD_EPS && previousHitDistForTracking > NRD_EPS )
            {
                float motion = Math::SmoothStep( 0.5, 6.0, smbParallaxInPixelsMax );
                float previousLogHitDist = log2( previousHitDistForTracking );
                float currentLogHitDist = log2( hitDistForTracking );
                float temporalFrameScale = 2.0 / max( gFramerateScale, 1.0 );
                float maxLogStep = lerp( 0.75, 0.30, motion ) * temporalFrameScale;
                float boundedLogHitDist = clamp( currentLogHitDist, previousLogHitDist - maxLogStep, previousLogHitDist + maxLogStep );
                float currentWeight = lerp( 0.55, 0.35, motion );
                currentWeight = 1.0 - pow( 1.0 - currentWeight, temporalFrameScale );
                float trackedLogHitDist = lerp( previousLogHitDist, boundedLogHitDist, currentWeight );
                currentLogHitDist = lerp( currentLogHitDist, trackedLogHitDist, smbFootprintQuality );
                hitDistForTracking = exp2( currentLogHitDist );
                gOut_SpecHitDistForTracking[ pixelPos ] = hitDistForTracking;
            }
        }
    #endif

    // Checkerboard resolve
    uint checkerboard = Sequence::CheckerBoard( pixelPos, gFrameIndex );
    #if( NRD_MODE == OCCLUSION )
        int3 checkerboardPos = pixelPos.xxy + int3( -1, 1, 0 );
        checkerboardPos.x = max( checkerboardPos.x, 0 );
        checkerboardPos.y = min( checkerboardPos.y, gRectSizeMinusOne.x );
        float viewZ0 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos.xz ) ] );
        float viewZ1 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos.yz ) ] );
        float disocclusionThresholdCheckerboard = GetDisocclusionThreshold( NRD_DISOCCLUSION_THRESHOLD, frustumSize, NoV );
        float2 wc = GetDisocclusionWeight( float2( viewZ0, viewZ1 ), viewZ, disocclusionThresholdCheckerboard );
        wc.x = ( !( viewZ0 < gDenoisingRange ) || pixelPos.x < 1 ) ? 0.0 : wc.x;
        wc.y = ( !( viewZ1 < gDenoisingRange ) || pixelPos.x >= gRectSizeMinusOne.x ) ? 0.0 : wc.y;
        wc *= Math::PositiveRcp( wc.x + wc.y );
        checkerboardPos.xy >>= 1;
    #endif

    // Specular
    #if( NRD_SPEC )
        // Accumulation speed
        float smbSpecHistoryLimit = min(gMaxAccumulatedFrameNum,
            max(gHistoryFixFrameNum + 1.0, gMaxAccumulatedFrameNum * pow(smbFootprintQuality / sizeQuality, 4.0)));
        smbSpecAccumSpeed = min(smbSpecAccumSpeed, smbSpecHistoryLimit);
        float smbSpecHistoryConfidence = sizeQuality;
        if( gHasHistoryConfidence && NRD_SUPPORTS_HISTORY_CONFIDENCE )
        {
            float confidence = saturate( gIn_SpecConfidence[ WithRectOrigin( pixelPos ) ] );
            smbSpecHistoryConfidence = min( smbSpecHistoryConfidence, confidence );
        }
        smbSpecAccumSpeed *= lerp( smbSpecHistoryConfidence, 1.0, 1.0 / ( 1.0 + smbSpecAccumSpeed ) );

        // Current
        bool specHasData = NRD_SUPPORTS_CHECKERBOARD == 0 || gSpecCheckerboard == 2 || checkerboard == gSpecCheckerboard;
        uint2 specPos = pixelPos;
        #if( NRD_MODE == OCCLUSION )
            specPos.x >>= gSpecCheckerboard == 2 ? 0 : 1;
        #endif

        REBLUR_TYPE spec = gIn_Spec[ specPos ];
        float specCurrentSpatialConfidence = 0.0;

        #if( NRD_MODE != OCCLUSION )
            if( gEnableLowRoughnessSpecularStabilization != 0 && opaqueLowRoughness )
            {
                float3 spatialSpec = spec.xyz;
                float spatialWeightSum = 1.0;
                float centerLuma = max( GetLuma( spec ), 0.0 );
                float spatialLumaM1 = centerLuma;
                float spatialLumaM2 = centerLuma * centerLuma;
                float spatialLumaSupport = 1.0;

                [unroll]
                for( j = -1; j <= 1; j++ )
                {
                    [unroll]
                    for( i = -1; i <= 1; i++ )
                    {
                        if( i == 0 && j == 0 )
                            continue;

                        int2 pos = clamp( int2( pixelPos ) + int2( i, j ), 0, gRectSizeMinusOne );
                        float w = GetLowRoughnessSpatialWeight( pos, materialID, N, roughness, viewZ );
                        REBLUR_TYPE s = gIn_Spec[ pos ];
                        float sampleLuma = max( GetLuma( s ), 0.0 );
                        spatialSpec += s.xyz * w;
                        spatialWeightSum += w;
                        spatialLumaM1 += sampleLuma * w;
                        spatialLumaM2 += sampleLuma * sampleLuma * w;
                        spatialLumaSupport += w * float(
                            sampleLuma + 5e-5 >= centerLuma * 0.35 + 5e-5 );
                    }
                }

                spec.xyz = spatialSpec / spatialWeightSum;
                spatialLumaM1 /= spatialWeightSum;
                spatialLumaM2 /= spatialWeightSum;
                float spatialLumaSigma = sqrt( max(
                    spatialLumaM2 - spatialLumaM1 * spatialLumaM1, 0.0 ) );
                float relativeSigma = spatialLumaSigma /
                    max( spatialLumaM1, 5e-5 );
                float centerAgreement =
                    ( min( centerLuma, spatialLumaM1 ) + 5e-5 ) /
                    ( max( centerLuma, spatialLumaM1 ) + 5e-5 );
                float spatialReliability = Math::SmoothStep(
                    4.0, 7.0, spatialWeightSum );
                float spatialSupport = Math::SmoothStep(
                    0.50, 0.80, spatialLumaSupport / spatialWeightSum );
                float spatialUniformity = 1.0 - Math::SmoothStep(
                    0.65, 1.50, relativeSigma );

                specCurrentSpatialConfidence = spatialReliability *
                    spatialSupport * spatialUniformity *
                    Math::SmoothStep( 0.25, 0.65, centerAgreement );
            }
        #endif

        // Checkerboard resolve // TODO: materialID support?
        #if( NRD_MODE == OCCLUSION )
            if( !specHasData )
            {
                float s0 = gIn_Spec[ checkerboardPos.xz ];
                float s1 = gIn_Spec[ checkerboardPos.yz ];

                s0 = Denanify( wc.x, s0 );
                s1 = Denanify( wc.y, s1 );

                spec = s0 * wc.x + s1 * wc.y;
            }
        #endif

        // Curvature estimation along predicted motion ( tests 15, 40, 76, 133, 146, 147, 148 )
        /*
        TODO: curvature! (-_-)
         - by design: curvature = 0 on static objects if camera is static
         - quantization errors hurt
         - curvature on bumpy surfaces is just wrong, pulling virtual positions into a surface and introducing lags
         - suboptimal reprojection if curvature changes signs under motion
        */
        float curvature = 0.0;
        {
            // IMPORTANT: non-zero parallax on objects attached to the camera is needed
            // IMPORTANT: the direction of "deltaUv" is important ( test 1 )
            float2 uvForZeroParallax = gOrthoMode == 0.0 ? smbPixelUv : pixelUv;
            float2 deltaUv = uvForZeroParallax - Geometry::GetScreenUv( gWorldToClipPrev, Xprev + gCameraDelta.xyz ); // TODO: repeats code for "smbParallaxInPixels1" with "-" sign
            deltaUv *= gRectSize;
            deltaUv /= max( smbParallaxInPixels1, 1.0 / 256.0 );

            // 10 edge
            float3 n10, x10;
            {
                float3 xv = Geometry::ReconstructViewPosition( pixelUv + float2( 1, 0 ) * gRectSizeInv, gFrustum, 1.0, gOrthoMode );
                float3 x = Geometry::RotateVector( gViewToWorld, xv );
                float3 v = GetViewVector( x );
                float3 o = gOrthoMode == 0.0 ? 0 : x;

                x10 = o + v * dot( X - o, N ) / dot( N, v ); // line-plane intersection
                n10 = s_Normal_HitDistForTracking[ threadPos.y + BORDER ][ threadPos.x + BORDER + 1 ].xyz;
            }

            // 01 edge
            float3 n01, x01;
            {
                float3 xv = Geometry::ReconstructViewPosition( pixelUv + float2( 0, 1 ) * gRectSizeInv, gFrustum, 1.0, gOrthoMode );
                float3 x = Geometry::RotateVector( gViewToWorld, xv );
                float3 v = GetViewVector( x );
                float3 o = gOrthoMode == 0.0 ? 0 : x;

                x01 = o + v * dot( X - o, N ) / dot( N, v ); // line-plane intersection
                n01 = s_Normal_HitDistForTracking[ threadPos.y + BORDER + 1 ][ threadPos.x + BORDER ].xyz;
            }

            // Mix
            float2 ww = abs( deltaUv ) + 1.0 / 256.0;
            ww /= ww.x + ww.y;

            float3 x = x10 * ww.x + x01 * ww.y;
            float3 n = normalize( n10 * ww.x + n01 * ww.y );

            // High parallax - flattens surface on high motion ( test 132, 172, 173, 174, 190, 201, 202, 203, e9 )
            // - "smbParallaxInPixelsMin" is used to get "0" ( ignore "high parallax" ) on objects attached to the camera
            // - increasing stride helps in corner cases due to better flattening, but on average it works worse ( test 1 if FPS <= 60 )
            float2 motionUvHigh = pixelUv + smbParallaxInPixelsMin * deltaUv * gRectSizeInv;

            // sqrt( 2.0 ) offers a smooth transition from one calculations to another without a hard border
            if( smbParallaxInPixelsMin > sqrt( 2.0 ) && IsInScreenNearest( motionUvHigh ) )
            {
                float2 uvScaled = WithRectOffset( ClampUvToViewport( motionUvHigh ) );

                float zHigh = UnpackViewZ( gIn_ViewZ.SampleLevel( gLinearClamp, uvScaled, 0 ) );
                float3 xHigh = Geometry::ReconstructViewPosition( motionUvHigh, gFrustum, zHigh, gOrthoMode );
                xHigh = Geometry::RotateVector( gViewToWorld, xHigh );

                float3 nHigh = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness.SampleLevel( STOCHASTIC_BILINEAR_FILTER, StochasticBilinear( uvScaled, gRectSize ), 0 ) ).xyz;

                // Replace if same surface
                float2 geometryWeightParams = GetGeometryWeightParams( NRD_CURVATURE_HIGH_PARALLAX_DISOCCLUSION_THRESHOLD, frustumSize, X, N, 1.0 );
                float NoX = dot( N, xHigh );

                float w = ComputeWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );
                w *= float( zHigh < gDenoisingRange );
                bool cmp = w > 0.5;

                n = cmp ? nHigh : n;
                x = cmp ? xHigh : x;
            }

            // Estimate curvature for the edge { x; X }
            float3 edge = x - X;
            float edgeLenSq = Math::LengthSquared( edge );
            curvature = dot( n - N, edge ) * Math::PositiveRcp( edgeLenSq );

            // Correction - very negative inconsistent with previous frame curvature blows up reprojection ( tests 164, 171 - 176 )
            if( curvature < 0 )
            {
                float2 uv1 = Geometry::GetScreenUv( gWorldToClipPrev, GetXvirtual( hitDistForTracking, curvature, X, X, N, V, roughness ) );
                float2 uv2 = Geometry::GetScreenUv( gWorldToClipPrev, X );
                float a = length( ( uv1 - uv2 ) * gRectSize );
                curvature *= float( a < NRD_MAX_ALLOWED_VIRTUAL_MOTION_ACCELERATION * smbParallaxInPixelsMax + gRectSizeInv.x );
            }
        }

        // Virtual motion - coordinates
        float3 Xvirtual = GetXvirtual( hitDistForTracking, curvature, X, Xprev, N, V, roughness );
        float XvirtualLength = length( Xvirtual );
        float hitDistanceToLobeSpreadInPixels = 1.0 / PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, XvirtualLength );

        float2 vmbPixelUv = Geometry::GetScreenUv( gWorldToClipPrev, Xvirtual );
        vmbPixelUv = materialID == gCameraAttachedReflectionMaterialID ? smbPixelUv : vmbPixelUv;

        float2 vmbDelta = vmbPixelUv - smbPixelUv;
        float vmbPixelsTraveled = length( vmbDelta * gRectSize );

        float2 vmbSampleUv = vmbPixelUv + gHistoryJitter;
        Filtering::Bilinear vmbBilinearFilter = Filtering::GetBilinearFilter( vmbSampleUv, gRectSizePrev );
        float2 vmbBilinearGatherUv = ( vmbBilinearFilter.origin + 1.0 ) * gResourceSizeInvPrev;

        float3 vmbNormal = N;
        #if( NRD_USE_PREV_WORLD_SPACE_MATRIX == 1 )
            vmbNormal = Geometry::RotateVectorInverse( gWorldPrevToWorld, vmbNormal );
        #endif

        int3 vmbOrigin = int3( vmbBilinearFilter.origin, 0 );
        float4 vmbNormal00 = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.Load( vmbOrigin ) );
        float4 vmbNormal10 = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.Load( vmbOrigin, int2( 1, 0 ) ) );
        float4 vmbNormal01 = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.Load( vmbOrigin, int2( 0, 1 ) ) );
        float4 vmbNormal11 = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.Load( vmbOrigin, int2( 1, 1 ) ) );
        float4 vmbNoN2x2 = float4( dot( vmbNormal00.xyz, vmbNormal ), dot( vmbNormal10.xyz, vmbNormal ),
            dot( vmbNormal01.xyz, vmbNormal ), dot( vmbNormal11.xyz, vmbNormal ) );
        float4 vmbRoughness = float4( vmbNormal00.w, vmbNormal10.w, vmbNormal01.w, vmbNormal11.w );
        float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams( roughness * roughness, gRoughnessFraction, REBLUR_ROUGHNESS_SENSITIVITY_IN_TA );
        float4 roughnessWeights = ComputeNonExponentialWeight( vmbRoughness * vmbRoughness, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y );
        roughnessWeights = lerp( 1.0, roughnessWeights, Math::SmoothStep01( vmbPixelsTraveled ) );
        float virtualHistoryConfidence;
        float4 vmbN;
        float vmbNoN;

        // Virtual motion - disocclusion
        float4 vmbOcclusionWeights;
        float vmbSpecAccumSpeed;
        bool vmbAllowCatRom;
        {
            // Disocclusion
            float4 vmbOcclusionThreshold = float4( vmbNoN2x2 > cosMaxAngle ); // normal // TODO: use lobe angle?
            vmbOcclusionThreshold *= step( 0.5, roughnessWeights ); // roughness
            vmbOcclusionThreshold *= IsInScreenBilinear( vmbBilinearFilter.origin, gRectSizePrev ); // in screen
            vmbOcclusionThreshold *= disocclusionThreshold * frustumSize;
            vmbOcclusionThreshold *= lerp( 0.1, 1.0, NoV ); // IMPORTANT: yes, "*" not "/"! This is a must for test 168 ( see contact shadow behind the heating radiator ), without this rare bright samples may stretch
            vmbOcclusionThreshold -= NRD_EPS;

            float4 vmbViewZ = UnpackViewZ( gPrev_ViewZ.GatherRed( gNearestClamp, vmbBilinearGatherUv ).wzxy );
            float3 vmbVv = Geometry::ReconstructViewPosition( vmbPixelUv, gFrustumPrev, 1.0 ); // unnormalized, orthoMode = 0
            float3 Nv = Geometry::RotateVector( gWorldToViewPrev, N );
            float NoXcurr = dot( N, Xprev - gCameraDelta.xyz );
            float4 NoXprev = ( Nv.x * vmbVv.x + Nv.y * vmbVv.y ) * ( gOrthoMode == 0 ? vmbViewZ : gOrthoMode ) + Nv.z * vmbVv.z * vmbViewZ;
            float4 vmbPlaneDist = abs( NoXprev - NoXcurr );

            float4 vmbOcclusion = step( vmbPlaneDist, vmbOcclusionThreshold ) * ( vmbViewZ < gDenoisingRange );

            // Prev data
            uint4 vmbInternalData = gPrev_InternalData.GatherRed( gNearestClamp, vmbBilinearGatherUv ).wzxy;

            float3 vmbInternalData00 = UnpackInternalData( vmbInternalData.x );
            float3 vmbInternalData10 = UnpackInternalData( vmbInternalData.y );
            float3 vmbInternalData01 = UnpackInternalData( vmbInternalData.z );
            float3 vmbInternalData11 = UnpackInternalData( vmbInternalData.w );

            #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
                // Disocclusion: material ID
                float4 vmbMaterialID = float4( vmbInternalData00.z, vmbInternalData10.z, vmbInternalData01.z, vmbInternalData11.z  );
                vmbOcclusion *= CompareMaterials( materialID, vmbMaterialID, gSpecMinMaterial );
            #endif

            // Save disocclusion bits
            fbits += vmbOcclusion.x * 16.0;
            fbits += vmbOcclusion.y * 32.0;
            fbits += vmbOcclusion.z * 64.0;
            fbits += vmbOcclusion.w * 128.0;

            // Accumulation speed
            vmbOcclusionWeights = Filtering::GetBilinearCustomWeights( vmbBilinearFilter, vmbOcclusion );
            vmbN = Filtering::ApplyBilinearCustomWeights( vmbNormal00, vmbNormal10, vmbNormal01, vmbNormal11, vmbOcclusionWeights );
            vmbN.xyz = _NRD_SafeNormalize( vmbN.xyz );
            #if( NRD_USE_PREV_WORLD_SPACE_MATRIX == 1 )
                vmbN.xyz = Geometry::RotateVector( gWorldPrevToWorld, vmbN.xyz );
            #endif
            vmbNoN = Filtering::ApplyBilinearCustomWeights( vmbNoN2x2.x, vmbNoN2x2.y, vmbNoN2x2.z, vmbNoN2x2.w, vmbOcclusionWeights );
            virtualHistoryConfidence = Filtering::ApplyBilinearCustomWeights( roughnessWeights.x, roughnessWeights.y, roughnessWeights.z, roughnessWeights.w, vmbOcclusionWeights );
            vmbSpecAccumSpeed = Filtering::ApplyBilinearCustomWeights( vmbInternalData00.y, vmbInternalData10.y, vmbInternalData01.y, vmbInternalData11.y, vmbOcclusionWeights );

            float vmbFootprintQuality = Filtering::ApplyBilinearFilter( vmbOcclusion.x, vmbOcclusion.y, vmbOcclusion.z, vmbOcclusion.w, vmbBilinearFilter );
            vmbFootprintQuality = Math::Sqrt01( vmbFootprintQuality );

            float vmbHistoryLimit = min(gMaxAccumulatedFrameNum,
                max(gHistoryFixFrameNum + 1.0, gMaxAccumulatedFrameNum * pow(vmbFootprintQuality, 4.0)));
            vmbSpecAccumSpeed = min(vmbSpecAccumSpeed, vmbHistoryLimit);
            float vmbSpecHistoryConfidence = 1.0;
            if( gHasHistoryConfidence && NRD_SUPPORTS_HISTORY_CONFIDENCE )
            {
                float confidence = saturate( gIn_SpecConfidence[ WithRectOrigin( pixelPos ) ] );
                vmbSpecHistoryConfidence = min( vmbSpecHistoryConfidence, confidence );
            }
            vmbSpecAccumSpeed *= lerp( vmbSpecHistoryConfidence, 1.0, 1.0 / ( 1.0 + vmbSpecAccumSpeed ) );

            // Is CatRom allowed? ( requires complete "vmbOcclusion" )
            vmbAllowCatRom = false;
        }

        // Estimate how many pixels are traveled by virtual motion - how many radians can it be?
        float curvatureAngle;
        float lobeHalfAngle;
        {
            // IMPORTANT: if curvature angle is multiplied by path length then we can get an angle exceeding "2 * PI", what is impossible.
            // The max angle is PI ( most left and most right points on a hemisphere ), it can be achieved by using "tan" instead of angle.
            float curvatureAngleTan = pixelSize * abs( curvature ); // tana = pixelSize / curvatureRadius = pixelSize * curvature
            curvatureAngleTan *= max( vmbPixelsTraveled / max( NoV, 0.01 ), 1.0 ); // path length
            curvatureAngleTan *= 2.0; // TODO: why it's here? but works well

            curvatureAngle = atan( curvatureAngleTan );

            // Copied from "GetNormalWeightParam" but doesn't use "lobeAngleFraction"
            float percentOfVolume = NRD_MAX_PERCENT_OF_LOBE_VOLUME / ( 1.0 + vmbSpecAccumSpeed );
            float lobeTanHalfAngle = ImportanceSampling::GetSpecularLobeTanHalfAngle( roughness, percentOfVolume );

            // TODO: use old code and sync with "GetNormalWeightParam"?
            //float lobeTanHalfAngle = ImportanceSampling::GetSpecularLobeTanHalfAngle( roughness, NRD_MAX_PERCENT_OF_LOBE_VOLUME );
            //lobeTanHalfAngle /= 1.0 + vmbSpecAccumSpeed;

            lobeTanHalfAngle = max( lobeTanHalfAngle, NRD_NORMAL_ENCODING_ERROR );
            hitDistanceToLobeSpreadInPixels *= lobeTanHalfAngle;

            lobeHalfAngle = atan( lobeTanHalfAngle );
        }

        // Virtual motion - confidence: parallax
        // Tests 3, 6, 8, 11, 14, 100, 103, 104, 106, 109, 110, 114, 120, 127, 130, 131, 132, 138, 139 and 9e
        float parallaxWeight;
        {
            float hitDistForTrackingPrev = gPrev_SpecHitDistForTracking.SampleLevel( gLinearClamp, vmbSampleUv * gResolutionScalePrev, 0 );
            float3 XvirtualPrev = GetXvirtual( hitDistForTrackingPrev, curvature, X, Xprev, N, V, roughness );

            float2 vmbPixelUvPrev = Geometry::GetScreenUv( gWorldToClipPrev, XvirtualPrev );
            vmbPixelUvPrev = materialID == gCameraAttachedReflectionMaterialID ? smbPixelUv : vmbPixelUvPrev;

            float r = min( hitDistForTracking, hitDistForTrackingPrev ) * hitDistanceToLobeSpreadInPixels;
            r *= 0.5; // strengthen the test
            r = max( r, 0.1 * roughness ); // clean up dirt for high roughness

            float d = length( ( vmbPixelUvPrev - vmbPixelUv ) * gRectSize );

            parallaxWeight = Math::LinearStep( r, 0.0, d );

        }

        // Virtual motion - confidence: normal
        {
            // TODO: is it needed? "vmbN" suffers from reprojection stretching...
            float normalWeight = GetEncodingAwareNormalWeight( N, vmbN.xyz, lobeHalfAngle, curvatureAngle, REBLUR_NORMAL_ULP );
            normalWeight = lerp( 1.0, normalWeight, Math::SmoothStep01( vmbPixelsTraveled ) ); // jitter friendly

            virtualHistoryConfidence *= normalWeight;
        }

        // Virtual motion - confidence: prev-prev tests
        {
            // IMPORTANT: 2 is needed because:
            // - line *** allows fallback to laggy surface motion, which can be wrongly redistributed by virtual motion
            // - we use at least linear filters, as the result a wider initial offset is needed
            float stepBetweenTaps = min( vmbPixelsTraveled * gFramerateScale, 2.0 ) + vmbPixelsTraveled / REBLUR_VIRTUAL_MOTION_PREV_PREV_WEIGHT_ITERATION_NUM;
            vmbDelta *= Math::Rsqrt( Math::LengthSquared( vmbDelta ) );
            vmbDelta /= gRectSizePrev;

            float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams( vmbN.w * vmbN.w, gRoughnessFraction, REBLUR_ROUGHNESS_SENSITIVITY_IN_TA ); // TODO: GetRoughnessWeightParams with 0.05 sensitivity?

            [unroll]
            for( i = 1; i <= REBLUR_VIRTUAL_MOTION_PREV_PREV_WEIGHT_ITERATION_NUM; i++ )
            {
                float2 vmbPixelUvPrev = vmbPixelUv + vmbDelta * i * stepBetweenTaps;
                float4 vmbNormalAndRoughnessPrev = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.SampleLevel( STOCHASTIC_BILINEAR_FILTER, StochasticBilinear( vmbPixelUvPrev + gHistoryJitter, gRectSizePrev ) * gResolutionScalePrev, 0 ) );

                #if( NRD_USE_PREV_WORLD_SPACE_MATRIX == 1 )
                    vmbNormalAndRoughnessPrev.xyz = Geometry::RotateVector( gWorldPrevToWorld, vmbNormalAndRoughnessPrev.xyz ); // from "prev" world space
                #endif

                float w = GetEncodingAwareNormalWeight( vmbN.xyz, vmbNormalAndRoughnessPrev.xyz, lobeHalfAngle, curvatureAngle * ( 1.0 + i * stepBetweenTaps ), REBLUR_NORMAL_ULP );
                w *= ComputeNonExponentialWeight( vmbNormalAndRoughnessPrev.w * vmbNormalAndRoughnessPrev.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y );

                #if( REBLUR_USE_STF == 1 && NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
                    // Cures issues of "StochasticBilinear" and produces closer look to the linear filter
                    w = lerp( 1.0, w, saturate( stepBetweenTaps ) );
                #endif

                w = IsInScreenNearest( vmbPixelUvPrev ) ? w : 1.0;

                // For "min" usage "virtualHistoryConfidence" must include only "roughness" and "normal" weights before this line
                virtualHistoryConfidence = min( virtualHistoryConfidence, w );
            }
        }

        // Virtual motion - confidence: apply parallax weight
        virtualHistoryConfidence *= parallaxWeight;

        // Surface history confidence ( test 9, 9e )
        // IMPORTANT: needs to be responsive, because "vmb" fails on bumpy surfaces for the following reasons:
        //  - normal and prev-prev tests fail
        //  - curvature is so high that "vmb" regresses to "smb" and starts to lag
        float surfaceHistoryConfidence;
        {
            float a = atan( smbParallaxInPixelsMax * pixelSize / length( X ) );
            //a = acos( saturate( dot( V, smbVprev ) ) ); // numerically unstable

            float nonLinearAccumSpeed = 1.0 / ( 1.0 + smbSpecAccumSpeed );
            float hPrev = ExtractHitDist( gHistory_Spec.SampleLevel( gLinearClamp, smbSampleUv * gResolutionScalePrev, 0 ) );
            float h = lerp( hPrev, ExtractHitDist( spec ), nonLinearAccumSpeed ) * hitDistNormalization;

            float tana0 = ImportanceSampling::GetSpecularLobeTanHalfAngle( roughnessModified, NRD_MAX_PERCENT_OF_LOBE_VOLUME ); // base lobe angle
            tana0 *= lerp( NoV, 1.0, roughnessModified ); // make more strict if NoV is low and lobe is very V-dependent
            tana0 *= nonLinearAccumSpeed; // make more strict if history is long
            tana0 /= GetHitDistFactor( h, frustumSize ) + NRD_EPS; // make relaxed "in corners", where reflection is close to the surface

            float a0 = max( atan( tana0 ), NRD_NORMAL_ENCODING_ERROR );

            float f = Math::LinearStep( a0, 0.0, a );
            surfaceHistoryConfidence = Math::Pow01( f, 4.0 );

            f = Math::LinearStep( 0.8, 0.9, roughnessModified );
            surfaceHistoryConfidence = lerp( surfaceHistoryConfidence, 1.0, f );
        }

        float lowRoughnessSurfaceFallback = 0.0;
        if( gEnableLowRoughnessSpecularStabilization != 0 && lowRoughnessSurfaceGuide && roughness <= guidedSpecularRoughnessLimit )
        {
            float motion = Math::SmoothStep( 1.0, 6.0, smbParallaxInPixelsMax );
            float vmbFailure = Math::SmoothStep( 0.15, 0.80, 1.0 - virtualHistoryConfidence );
            lowRoughnessSurfaceFallback = motion * vmbFailure;

            float fallbackFrameNum = materialID > 1.5 ? 10.0 : 6.0;
            surfaceHistoryConfidence = max( surfaceHistoryConfidence,
                lowRoughnessSurfaceFallback * ( fallbackFrameNum / max( gMaxAccumulatedFrameNum, 1.0 ) ) );
        }

        // Limit number of accumulated frames
        float smbSpecAccumSpeed_NoHistoryFix;
        float vmbSpecAccumSpeed_NoHistoryFix;
        {
            // Responsive accumulation
            // Use "roughnessModified" to bring some "AA" goodness
            float responsiveFactor = RemapRoughnessToResponsiveFactor( roughnessModified );
            float smc = GetSpecMagicCurve( roughnessModified );

            float2 f;
            f.x = smbNoN;
            f.y = vmbNoN;
            f = lerp( smc, 1.0, responsiveFactor ) * Math::Pow01( f, lerp( 32.0, 1.0, smc ) * ( 1.0 - responsiveFactor ) );

            float2 maxResponsiveFrameNum = gMaxAccumulatedFrameNum;
            maxResponsiveFrameNum *= f;
            maxResponsiveFrameNum = max( maxResponsiveFrameNum, gResponsiveAccumulationMinAccumulatedFrameNum );

            // Apply limits
            float2 maxFrameNum = gMaxAccumulatedFrameNum * float2( surfaceHistoryConfidence, virtualHistoryConfidence );
            float2 maxFrameNum_NoHistoryFix = min( maxFrameNum, max( maxResponsiveFrameNum, gHistoryFixFrameNum ) );

            smbSpecAccumSpeed_NoHistoryFix = min( smbSpecAccumSpeed, maxFrameNum_NoHistoryFix.x );
            vmbSpecAccumSpeed_NoHistoryFix = min( vmbSpecAccumSpeed, maxFrameNum_NoHistoryFix.y );

            maxFrameNum = min( maxFrameNum, maxResponsiveFrameNum );

            smbSpecAccumSpeed = min( smbSpecAccumSpeed, maxFrameNum.x );
            vmbSpecAccumSpeed = min( vmbSpecAccumSpeed, maxFrameNum.y );
        }

        // Virtual history amount ( tests 65, 66, 103, 107, 111, 132, e9, e11, 218 ) // ***
        // OLD: virtualHistoryAmount = saturate( scale )
        //      * Dfactor                   - 1 is assumed now, because "Dfactor" is applied in "GetXvirtual" to "vmbPixelUv" making it closer to surface where needed ( test 236 )
        //    Helped on bumpy surfaces, because virtual motion got ruined by big curvature
        //      * normalBasedConfidence     - 1 is assumed now, because the selector below does the same and avoids "double applying" ( was used before "prev-prev" test )
        //    Helped to preserve "lying-on-surface" roughness details
        //      * roughnessBasedConfidence  - 1 is assumed now, because the selector below does the same and avoids "double applying"
        float virtualHistoryAmount;
        {
            // "1" if "vmb" >= "smb", pull towards "smb" based on delta otherwise
            virtualHistoryAmount = 1.0 + ( vmbSpecAccumSpeed - smbSpecAccumSpeed ) / ( 1.0 + 0.5 * max( vmbSpecAccumSpeed, smbSpecAccumSpeed ) ); // TODO: 0.5 => 0.25?
            virtualHistoryAmount = saturate( virtualHistoryAmount );

            // - dithering is not needed, since "vmb" is dominating for any possible "roughness, NoV"
            // - choose only one if the other one is not-fully valid
            if( !smbAllowCatRom || !vmbAllowCatRom ) // TODO: doing "step" unconditionally is the safest approach
                virtualHistoryAmount = step( 0.5, virtualHistoryAmount );
        }

        if( lowRoughnessSurfaceFallback > 0.0 )
            virtualHistoryAmount = lerp( virtualHistoryAmount, 0.0, lowRoughnessSurfaceFallback );

        // Sample history
        REBLUR_TYPE specHistory;
        REBLUR_FAST_TYPE specFastHistory;
        REBLUR_SH_TYPE specShHistory;
        {
            float2 uv = lerp( smbSampleUv, vmbSampleUv, virtualHistoryAmount );
            float4 occlusionWeights = lerp( smbOcclusionWeights, vmbOcclusionWeights, virtualHistoryAmount );
            bool allowCatRom = virtualHistoryAmount < 0.5 ? smbAllowCatRom : vmbAllowCatRom;

            BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
                saturate( uv ) * gRectSizePrev, gResourceSizeInvPrev,
                occlusionWeights, allowCatRom,
                gHistory_Spec, specHistory,
                gHistory_SpecFast, specFastHistory
                #if( NRD_MODE == SH )
                    , gHistory_SpecSh, specShHistory
                #endif
            );

            // Avoid negative values
            specHistory = ClampNegativeToZero( specHistory );
            specFastHistory = max( specFastHistory, 0.0 );
        }

        // Accumulation
        float specAccumSpeedCorrected = lerp( smbSpecAccumSpeed_NoHistoryFix, vmbSpecAccumSpeed_NoHistoryFix, virtualHistoryAmount ); // avoid "HistoryFix" in responsive accumulation
        float specAccumSpeed = lerp( smbSpecAccumSpeed, vmbSpecAccumSpeed, virtualHistoryAmount );
        float specNonLinearAccumSpeed = 1.0 / ( 1.0 + specAccumSpeed );
        float specCoherentChangeConfidence = 0.0;

        if( !specHasData )
            specNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, specNonLinearAccumSpeed );

        #if( NRD_MODE != OCCLUSION )
            if( gEnableLowRoughnessSpecularStabilization != 0 &&
                opaqueLowRoughness && specHasData )
            {
                float currentLuma = max( GetLuma( spec ), 0.0 );
                float historyLuma = max( GetLuma( specHistory ), 0.0 );
                float relativeChange = abs( currentLuma - historyLuma ) /
                    max( max( currentLuma, historyLuma ), 0.001 );
                specCoherentChangeConfidence = specCurrentSpatialConfidence *
                    Math::SmoothStep( 0.10, 0.30, relativeChange );

                float baseResponse = currentLuma < historyLuma ? 0.42 : 0.18;
                float temporalFrameScale = 2.0 / max( gFramerateScale, 1.0 );
                float coherentResponse = 1.0 - pow(
                    1.0 - baseResponse, temporalFrameScale );
                specNonLinearAccumSpeed = max(
                    specNonLinearAccumSpeed,
                    coherentResponse * specCoherentChangeConfidence );
            }
        #endif

        bool specMissing = !specHasData && !any( spec != 0.0 ) && specAccumSpeed > 0.0;
        specNonLinearAccumSpeed *= float( !specMissing );
        REBLUR_TYPE specResult = MixHistoryAndCurrent( specHistory, spec, specNonLinearAccumSpeed, roughness );

        #if( NRD_MODE == SH )
            REBLUR_SH_TYPE specSh = gIn_SpecSh[ specPos ];
            if( gEnableLowRoughnessSpecularStabilization != 0 && opaqueLowRoughness )
            {
                float3 spatialSpecSh = specSh.xyz;
                float spatialWeightSum = 1.0;

                [unroll]
                for( j = -1; j <= 1; j++ )
                {
                    [unroll]
                    for( i = -1; i <= 1; i++ )
                    {
                        if( i == 0 && j == 0 )
                            continue;

                        int2 pos = clamp( int2( pixelPos ) + int2( i, j ), 0, gRectSizeMinusOne );
                        float w = GetLowRoughnessSpatialWeight( pos, materialID, N, roughness, viewZ );
                        spatialSpecSh += gIn_SpecSh[ pos ].xyz * w;
                        spatialWeightSum += w;
                    }
                }

                specSh.xyz = spatialSpecSh / spatialWeightSum;
            }
            REBLUR_SH_TYPE specShResult = lerp( specShHistory, specSh, specNonLinearAccumSpeed );
        #endif

        // Firefly suppressor
        float specMaxRelativeIntensity = gFireflySuppressorMinRelativeScale + REBLUR_FIREFLY_SUPPRESSOR_MAX_RELATIVE_INTENSITY / ( specAccumSpeed + 1.0 );

        float specAntifireflyFactor = specAccumSpeed * gMaxBlurRadius * REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE;
        specAntifireflyFactor /= 1.0 + specAntifireflyFactor;

        #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
        {
            float specLumaResult = GetLuma( specResult );
            float specFireflyUpper = GetLuma( specHistory ) * specMaxRelativeIntensity;
            float currentLuma = max( GetLuma( spec ), 0.0 );
            float coherentRiseConfidence = specCoherentChangeConfidence *
                float( currentLuma > GetLuma( specHistory ) );
            specFireflyUpper = max(
                specFireflyUpper,
                lerp( specFireflyUpper, currentLuma, coherentRiseConfidence ) );
            float specLumaClamped = min( specLumaResult, specFireflyUpper );
            specLumaClamped = lerp( specLumaResult, specLumaClamped, specAntifireflyFactor );

            specResult = ChangeLuma( specResult, specLumaClamped );
            #if( NRD_MODE == SH )
                specShResult *= GetLumaScale( length( specShResult ), specLumaClamped );
            #endif

            float specHitDistMaxRelativeIntensity = 1.2 + 1.0 / ( specAccumSpeed + 1.0 );
            specResult.w = lerp( specResult.w, min( specResult.w, specHistory.w * specHitDistMaxRelativeIntensity ), specAntifireflyFactor );
        }
        #endif

        // Output
        gOut_Spec[ pixelPos ] = specResult;
        #if( NRD_MODE == SH )
            gOut_SpecSh[ pixelPos ] = specShResult;
        #endif

        { // Fast history
            float maxFastAccumulatedFrameNum = gMaxFastAccumulatedFrameNum;
            if( materialID == gStrandMaterialID )
                maxFastAccumulatedFrameNum = max( maxFastAccumulatedFrameNum, gMaxAccumulatedFrameNum / 5 );

            float specHistoryConfidence = lerp( surfaceHistoryConfidence, virtualHistoryConfidence, virtualHistoryAmount );
            float specFastNonLinearAccumSpeed = GetNonLinearAccumSpeed( specAccumSpeed, maxFastAccumulatedFrameNum, specHistoryConfidence, specHasData );
            specFastNonLinearAccumSpeed *= float( !specMissing );
            float specFastResult = lerp( specFastHistory, GetLuma( spec ), specFastNonLinearAccumSpeed );

            // Firefly suppressor ( fixes heavy crawling under camera rotation: test 95, 120 )
            #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
                float specFastUpper = GetLuma( specHistory ) *
                    specMaxRelativeIntensity *
                    REBLUR_FIREFLY_SUPPRESSOR_FAST_RELATIVE_INTENSITY;
                float currentLuma = max( GetLuma( spec ), 0.0 );
                float coherentRiseConfidence = specCoherentChangeConfidence *
                    float( currentLuma > GetLuma( specHistory ) );
                specFastUpper = max(
                    specFastUpper,
                    lerp( specFastUpper, currentLuma, coherentRiseConfidence ) );
                float specFastClamped = min( specFastResult, specFastUpper );
                specFastResult = lerp( specFastResult, specFastClamped, specAntifireflyFactor );
            #endif

            gOut_SpecFast[ pixelPos ] = specFastResult;
        }

        // Debug
        #if( REBLUR_SHOW == REBLUR_SHOW_CURVATURE )
            virtualHistoryAmount = abs( curvature ) * pixelSize * 30.0;
        #elif( REBLUR_SHOW == REBLUR_SHOW_CURVATURE_SIGN )
            virtualHistoryAmount = sign( curvature ) * 0.5 + 0.5;
        #elif( REBLUR_SHOW == REBLUR_SHOW_SURFACE_HISTORY_CONFIDENCE )
            virtualHistoryAmount = surfaceHistoryConfidence;
        #elif( REBLUR_SHOW == REBLUR_SHOW_VIRTUAL_HISTORY_CONFIDENCE )
            virtualHistoryAmount = virtualHistoryConfidence;
        #elif( REBLUR_SHOW == REBLUR_SHOW_HIT_DIST_FOR_TRACKING )
            float smc = GetSpecMagicCurve( roughness );
            virtualHistoryAmount = hitDistForTracking * lerp( 1.0, 5.0, smc ) / ( 1.0 + hitDistForTracking * lerp( 1.0, 5.0, smc ) );
        #endif
    #else
        float specAccumSpeedCorrected = 0;
        float curvature = 0;
        float virtualHistoryAmount = 0;
    #endif

    // Output
    #if( NRD_MODE != OCCLUSION )
        // TODO: "PackData2" can be inlined into the code ( right after a variable gets ready for use ) to utilize the only
        // one "uint" for the intermediate storage. But it looks like the compiler does good job by rearranging the code for us
        gOut_Data2[ pixelPos ] = PackData2( fbits, curvature, virtualHistoryAmount, smbAllowCatRom );
    #endif

    // Diffuse
    #if( NRD_DIFF )
        // Accumulation speed
        float diffHistoryLimit = min(gMaxAccumulatedFrameNum,
            max(gHistoryFixFrameNum + 1.0, gMaxAccumulatedFrameNum * pow(smbFootprintQuality / sizeQuality, 4.0)));
        diffAccumSpeed = min(diffAccumSpeed, diffHistoryLimit);
        float diffHistoryConfidence = sizeQuality;
        if( gHasHistoryConfidence && NRD_SUPPORTS_HISTORY_CONFIDENCE )
        {
            float confidence = saturate( gIn_DiffConfidence[ WithRectOrigin( pixelPos ) ] );
            diffHistoryConfidence = min( diffHistoryConfidence, confidence );
        }
        diffAccumSpeed *= lerp( diffHistoryConfidence, 1.0, 1.0 / ( 1.0 + diffAccumSpeed ) );
        diffAccumSpeed = min( diffAccumSpeed, gMaxAccumulatedFrameNum );

        // Current
        bool diffHasData = NRD_SUPPORTS_CHECKERBOARD == 0 || gDiffCheckerboard == 2 || checkerboard == gDiffCheckerboard;
        uint2 diffPos = pixelPos;
        #if( NRD_MODE == OCCLUSION )
            diffPos.x >>= gDiffCheckerboard == 2 ? 0 : 1;
        #endif

        REBLUR_TYPE diff = gIn_Diff[ diffPos ];

        // Checkerboard resolve // TODO: materialID support?
        #if( NRD_MODE == OCCLUSION )
            if( !diffHasData )
            {
                float d0 = gIn_Diff[ checkerboardPos.xz ];
                float d1 = gIn_Diff[ checkerboardPos.yz ];

                d0 = Denanify( wc.x, d0 );
                d1 = Denanify( wc.y, d1 );

                diff = d0 * wc.x + d1 * wc.y;
            }
        #endif

        #if( NRD_MODE == RADIANCE || NRD_MODE == SH )
            gOut_DiffCurrentLuma[ pixelPos ] = max( GetLuma( diff ), 0.0 );
        #endif

        // Sample history
        REBLUR_TYPE diffHistory;
        REBLUR_FAST_TYPE diffFastHistory;
        REBLUR_SH_TYPE diffShHistory;
        {
            BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
                saturate( smbSampleUv ) * gRectSizePrev, gResourceSizeInvPrev,
                smbOcclusionWeights, smbAllowCatRom,
                gHistory_Diff, diffHistory,
                gHistory_DiffFast, diffFastHistory
                #if( NRD_MODE == SH )
                    , gHistory_DiffSh, diffShHistory
                #endif
            );

            // Avoid negative values
            diffHistory = ClampNegativeToZero( diffHistory );
            diffFastHistory = max( diffFastHistory, 0.0 );
        }

        // Accumulation
        float diffNonLinearAccumSpeed = 1.0 / ( 1.0 + diffAccumSpeed );

        if( !diffHasData )
            diffNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, diffNonLinearAccumSpeed );

        #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
            float2 diffAdmittedNormalizedChroma = 0.0;
            bool adjustDiffChroma = false;
            bool useDiffChromaAdmission =
                gEnableLowRoughnessSpecularStabilization != 0 &&
                materialID < 0.5 && diffHasData;
            float diffTemporalFrameScale = 2.0 / max( gFramerateScale, 1.0 );
            if( useDiffChromaAdmission )
            {
                float diffHistoryLumaForChroma = max( GetLuma( diffHistory ), 0.0 );
                float diffCurrentLumaForChroma = max( GetLuma( diff ), 0.0 );
                float2 historyChroma = diffHistoryLumaForChroma > 1e-5 ?
                    diffHistory.yz / diffHistoryLumaForChroma : 0.0;
                float2 currentChroma = diffCurrentLumaForChroma > 1e-5 ?
                    diff.yz / diffCurrentLumaForChroma : 0.0;
                historyChroma = clamp( historyChroma, -4.0, 4.0 );
                currentChroma = clamp( currentChroma, -4.0, 4.0 );

                float historyChromaLengthSq = dot( historyChroma, historyChroma );
                float currentChromaLengthSq = dot( currentChroma, currentChroma );
                float chromaAlignment = dot( historyChroma, currentChroma );
                bool opposingChroma = historyChromaLengthSq > 1e-4 &&
                    currentChromaLengthSq > 1e-4 && chromaAlignment <= 0.0;
                adjustDiffChroma = opposingChroma;
                if( adjustDiffChroma )
                {
                    float2 chromaTarget = 0.0;
                    float baseChromaResponse = 0.85;
                    float chromaResponse = 1.0 - pow(
                        1.0 - baseChromaResponse, diffTemporalFrameScale );
                    diffAdmittedNormalizedChroma = lerp(
                        historyChroma, chromaTarget, chromaResponse );
                }
            }
        #endif

        bool diffMissing = !diffHasData && !any(diff != 0.0) && diffAccumSpeed > 0.0;
        diffNonLinearAccumSpeed *= float(!diffMissing);
        REBLUR_TYPE diffResult = MixHistoryAndCurrent( diffHistory, diff, diffNonLinearAccumSpeed );
        #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
            if( useDiffChromaAdmission && adjustDiffChroma )
                diffResult.yz = diffAdmittedNormalizedChroma *
                    max( GetLuma( diffResult ), 0.0 );
        #endif
        #if( NRD_MODE == SH )
            REBLUR_SH_TYPE diffSh = gIn_DiffSh[ diffPos ];
            REBLUR_SH_TYPE diffShResult = lerp( diffShHistory, diffSh, diffNonLinearAccumSpeed );
        #endif

        // Firefly suppressor
        #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
            float diffMaxRelativeIntensity = gFireflySuppressorMinRelativeScale + REBLUR_FIREFLY_SUPPRESSOR_MAX_RELATIVE_INTENSITY / ( diffAccumSpeed + 1.0 );

            float diffAntifireflyFactor = diffAccumSpeed * gMaxBlurRadius * REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE;
            diffAntifireflyFactor /= 1.0 + diffAntifireflyFactor;

            float diffLumaResult = GetLuma( diffResult );
            float diffHistoryLuma = GetLuma( diffHistory );
            float diffFireflyUpper = max( diffHistoryLuma * diffMaxRelativeIntensity,
                                          diffHistoryLuma + 0.016 * diffTemporalFrameScale );
            float diffLumaClamped = min( diffLumaResult, diffFireflyUpper );
            diffLumaClamped = lerp( diffLumaResult, diffLumaClamped, diffAntifireflyFactor );

            diffResult = ChangeLuma( diffResult, diffLumaClamped );
            #if( NRD_MODE == SH )
                diffShResult *= GetLumaScale( length( diffShResult ), diffLumaClamped );
            #endif

            float diffHitDistMaxRelativeIntensity = 1.2 + 1.0 / ( diffAccumSpeed + 1.0 );
            diffResult.w = lerp( diffResult.w, min( diffResult.w, diffHistory.w * diffHitDistMaxRelativeIntensity ), diffAntifireflyFactor );
        #endif

        // Output
        gOut_Diff[ pixelPos ] = diffResult;
        #if( NRD_MODE == SH )
            gOut_DiffSh[ pixelPos ] = diffShResult;
        #endif

        { // Fast history
            float diffFastAccumSpeed = min( diffAccumSpeed, gMaxFastAccumulatedFrameNum );
            float diffFastNonLinearAccumSpeed = 1.0 / ( 1.0 + diffFastAccumSpeed );

            if( !diffHasData )
                diffFastNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, diffFastNonLinearAccumSpeed );

            diffFastNonLinearAccumSpeed *= float(!diffMissing);
            float diffFastResult = lerp( diffFastHistory, GetLuma( diff ), diffFastNonLinearAccumSpeed );

            #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
                // Firefly suppressor ( fixes heavy crawling under camera rotation, test 99 )
                float diffFastUpper = max( diffHistoryLuma * diffMaxRelativeIntensity * REBLUR_FIREFLY_SUPPRESSOR_FAST_RELATIVE_INTENSITY,
                                           diffHistoryLuma + 0.016 * diffTemporalFrameScale );
                float diffFastClamped = min( diffFastResult, diffFastUpper );
                diffFastResult = lerp( diffFastResult, diffFastClamped, diffAntifireflyFactor );
            #endif

            gOut_DiffFast[ pixelPos ] = diffFastResult;
        }
    #else
        float diffAccumSpeed = 0;
    #endif

    // Output
    gOut_Data1[ pixelPos ] = PackData1( diffAccumSpeed, specAccumSpeedCorrected );
}
