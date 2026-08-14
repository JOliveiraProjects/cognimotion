# ─────────────────────────────────────────────────────────────────────────────
# cmi_worker — processo de inferência LibTorch isolado do Unreal.
#
# Linka a MESMA LibTorch que está em ../ThirdParty/LibTorch/LibTorch/
# (a que o plugin usa). Gera cmi_worker.exe.
#
# COMO COMPILAR (PowerShell, na pasta deste arquivo):
#   mkdir build
#   cd build
#   cmake -G "Visual Studio 17 2022" -A x64 ^
#         -DCMAKE_PREFIX_PATH="../../ThirdParty/LibTorch/LibTorch" ..
#   cmake --build . --config Release
#
# O .exe sai em build/Release/cmi_worker.exe. Copie-o para:
#   <Plugin>/Binaries/Win64/cmi_worker.exe
# (o plugin procura o worker ao lado dos binários do módulo).
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.18)
project(cmi_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Caminho para a LibTorch do plugin. Pode ser sobrescrito com -DCMAKE_PREFIX_PATH.
if(NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../ThirdParty/LibTorch/LibTorch")
endif()

find_package(Torch REQUIRED)

add_executable(cmi_worker cmi_worker.cpp)
target_link_libraries(cmi_worker "${TORCH_LIBRARIES}")
set_property(TARGET cmi_worker PROPERTY CXX_STANDARD 17)

# Release otimizado. NOMINMAX evita que <windows.h> defina macros min/max que
# colidem com a STL/torch. /EHsc habilita exceptions C++ (a torch usa).
if(MSVC)
    target_compile_options(cmi_worker PRIVATE /O2 /EHsc /bigobj)
    target_compile_definitions(cmi_worker PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# Copia as DLLs da LibTorch para junto do .exe (necessário em runtime).
if(MSVC)
    file(GLOB TORCH_DLLS "${CMAKE_PREFIX_PATH}/lib/*.dll")
    add_custom_command(TARGET cmi_worker POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:cmi_worker>)
endif()
