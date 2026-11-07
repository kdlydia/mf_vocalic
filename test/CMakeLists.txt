cmake_minimum_required(VERSION 3.25)

set(MF_MIN_VERSION "1.0.0")
set(MF_NEEDS_LILA OFF)
project(mf_vocalic_test LANGUAGES C CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(MAYAFLUX_PROJECT)

set(MAYAFLUX_SEARCH_PATHS "")

if(DEFINED ENV{MAYAFLUX_ROOT})
    list(APPEND MAYAFLUX_SEARCH_PATHS "$ENV{MAYAFLUX_ROOT}/lib/cmake/MayaFlux")
    list(APPEND MAYAFLUX_SEARCH_PATHS "$ENV{MAYAFLUX_ROOT}/lib64/cmake/MayaFlux")
elseif(APPLE)
    list(APPEND MAYAFLUX_SEARCH_PATHS "/opt/homebrew/lib/cmake/MayaFlux")
    list(APPEND MAYAFLUX_SEARCH_PATHS "/usr/local/lib/cmake/MayaFlux")
elseif(UNIX)
    list(APPEND MAYAFLUX_SEARCH_PATHS "/usr/lib/cmake/MayaFlux")
    list(APPEND MAYAFLUX_SEARCH_PATHS "/usr/lib64/cmake/MayaFlux")
    list(APPEND MAYAFLUX_SEARCH_PATHS "/usr/local/lib/cmake/MayaFlux")
elseif(WIN32)
    list(APPEND MAYAFLUX_SEARCH_PATHS "C:/MayaFlux/lib/cmake/MayaFlux")
endif()

find_package(MayaFlux REQUIRED PATHS ${MAYAFLUX_SEARCH_PATHS} NO_DEFAULT_PATH)

if(NOT EXISTS "${MayaFlux_PCH_FILE}")
    message(FATAL_ERROR
        "MayaFlux runtime PCH not found at ${MayaFlux_PCH_FILE}\n"
        "Ensure MayaFlux is correctly installed."
    )
endif()

file(GLOB_RECURSE _mf_vocalic_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp"
)

add_library(mf_vocalic OBJECT ${_mf_vocalic_sources})

set_target_properties(mf_vocalic PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(mf_vocalic PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../src")

target_precompile_headers(mf_vocalic PUBLIC "${MayaFlux_PCH_FILE}")
target_link_libraries(mf_vocalic PUBLIC MayaFlux::MayaFluxLib)

if(MF_NEEDS_LILA)
    target_link_libraries(mf_vocalic PUBLIC MayaFlux::MayaFluxHost)
endif()

add_executable(test_mf_vocalic 
    test_mf_vocalic.cpp
    $<TARGET_OBJECTS:mf_vocalic>
)

target_precompile_headers(test_mf_vocalic REUSE_FROM mf_vocalic)
target_link_libraries(test_mf_vocalic PRIVATE mf_vocalic MayaFlux::MayaFluxLib)

add_test(NAME mf_vocalic COMMAND test_mf_vocalic)

if(UNIX)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_CURRENT_SOURCE_DIR}/../compile_commands.json
    )
endif()
