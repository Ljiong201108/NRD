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
#include "REBLUR_PrePass.resources.hlsli"

#include "Common.hlsli"

#include "REBLUR_Common.hlsli"

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_REVERSED;

    // Tile-based early out
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // Early out
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pixelPos ) ] );
    if( viewZ > gDenoisingRange )
        return;

    // Normal and roughness
    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ], materialID );
    float3 N = normalAndRoughness.xyz;
    float3 Nv = Geometry::RotateVectorInverse( gViewToWorld, N );
    float roughness = normalAndRoughness.w;

    // Shared data
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );

    float3 Vv = GetViewVector( Xv, true );
    float NoV = abs( dot( Nv, Vv ) );

    const float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );
    const float4 rotator = GetBlurKernelRotation( REBLUR_PRE_BLUR_ROTATOR_MODE, pixelPos, gRotatorPre, gFrameIndex );

    // Checkerboard resolve
#if( NRD_SUPPORTS_CHECKERBOARD == 1 )
    uint checkerboard = Sequence::CheckerBoard( pixelPos, gFrameIndex );

    int2 checkerboardPos0 = int2( max( int( pixelPos.x ) - 1, 0 ), pixelPos.y );
    int2 checkerboardPos1 = int2( min( pixelPos.x + 1, gRectSizeMinusOne.x ), pixelPos.y );
    float viewZ0 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos0 ) ] );
    float viewZ1 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos1 ) ] );
    float disocclusionThresholdCheckerboard = GetDisocclusionThreshold( NRD_DISOCCLUSION_THRESHOLD, frustumSize, NoV );
    float2 wc = GetDisocclusionWeight( float2( viewZ0, viewZ1 ), viewZ, disocclusionThresholdCheckerboard );
    float3 normal0 = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( checkerboardPos0 ) ] ).xyz;
    float3 normal1 = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( checkerboardPos1 ) ] ).xyz;
    wc *= float2( dot( N, normal0 ) >= 0.5, dot( N, normal1 ) >= 0.5 );
    wc.x = ( viewZ0 > gDenoisingRange || pixelPos.x < 1 ) ? 0.0 : wc.x;
    wc.y = ( viewZ1 > gDenoisingRange || pixelPos.x >= gRectSizeMinusOne.x ) ? 0.0 : wc.y;
    if( !any( wc ) && ( ( gDiffCheckerboard != 2 && checkerboard != gDiffCheckerboard ) || ( gSpecCheckerboard != 2 && checkerboard != gSpecCheckerboard ) ) )
    {
        checkerboardPos0 = int2( pixelPos.x, max( int( pixelPos.y ) - 1, 0 ) );
        checkerboardPos1 = int2( pixelPos.x, min( pixelPos.y + 1, gRectSizeMinusOne.y ) );
        viewZ0 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos0 ) ] );
        viewZ1 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos1 ) ] );
        normal0 = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( checkerboardPos0 ) ] ).xyz;
        normal1 = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( checkerboardPos1 ) ] ).xyz;
        wc = GetDisocclusionWeight( float2( viewZ0, viewZ1 ), viewZ, disocclusionThresholdCheckerboard );
        wc *= float2( dot( N, normal0 ) >= 0.5, dot( N, normal1 ) >= 0.5 );
        wc.x = ( viewZ0 > gDenoisingRange || pixelPos.y < 1 ) ? 0.0 : wc.x;
        wc.y = ( viewZ1 > gDenoisingRange || pixelPos.y >= gRectSizeMinusOne.y ) ? 0.0 : wc.y;
    }
    wc *= Math::PositiveRcp( wc.x + wc.y );
    checkerboardPos0.x >>= 1;
    checkerboardPos1.x >>= 1;

#endif

    // Spatial filtering
    #define REBLUR_SPATIAL_MODE REBLUR_PRE_BLUR

    #if( NRD_DIFF )
    {
        uint2 pos = pixelPos;
        pos.x >>= gDiffCheckerboard == 2 ? 0 : 1;

        float sum = 1.0;
        REBLUR_TYPE diff = gIn_Diff[ pos ];
        #if( NRD_MODE == SH )
            float4 diffSh = gIn_DiffSh[ pos ];
        #endif

    #if( NRD_SUPPORTS_CHECKERBOARD == 1 )
        if( gDiffCheckerboard != 2 && checkerboard != gDiffCheckerboard )
        {
            sum = 0;
            diff = 0;
            #if( NRD_MODE == SH )
                diffSh = 0;
            #endif
        }
    #endif

        #include "REBLUR_Common_DiffuseSpatialFilter.hlsli"
    }
    #endif

    #if( NRD_SPEC )
    {
        uint2 pos = pixelPos;
        pos.x >>= gSpecCheckerboard == 2 ? 0 : 1;

        float sum = 1.0;
        REBLUR_TYPE spec = gIn_Spec[ pos ];
        #if( NRD_MODE == SH )
            float4 specSh = gIn_SpecSh[ pos ];
        #endif

    #if( NRD_SUPPORTS_CHECKERBOARD == 1 )
        if( gSpecCheckerboard != 2 && checkerboard != gSpecCheckerboard )
        {
            sum = 0;
            spec = 0;
            #if( NRD_MODE == SH )
                specSh = 0;
            #endif
        }
    #endif

        #include "REBLUR_Common_SpecularSpatialFilter.hlsli"
    }
    #endif
}
