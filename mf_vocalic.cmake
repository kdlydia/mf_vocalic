set(MF_MIN_VERSION "1.0.0")
set(MF_NEEDS_LILA OFF)

file(GLOB_RECURSE _mf_vocalic_sources CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/community/mf_vocalic/src/*.cpp"
)

add_library(mf_vocalic OBJECT
    ${_mf_vocalic_sources}
)

target_include_directories(mf_vocalic PUBLIC
    "${CMAKE_SOURCE_DIR}/community/mf_vocalic/src"
)

set_target_properties(mf_vocalic PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(mf_vocalic PUBLIC MayaFlux::MayaFluxLib)

if(MF_NEEDS_LILA)
    target_link_libraries(mf_vocalic PUBLIC MayaFlux::MayaFluxHost)
endif()
