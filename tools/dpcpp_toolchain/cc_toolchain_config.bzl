load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

all_compile_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.lto_backend,
    ACTION_NAMES.clif_match,
]

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "/opt/intel/oneapi/compiler/2025.1/bin/compiler/clang",
        ),
        tool_path(
            name = "ld",
            path = "/opt/intel/oneapi/compiler/2025.1/bin/compiler/llvm-link",
        ),
        tool_path(
            name = "ar",
            path = "/opt/intel/oneapi/compiler/2025.1/bin/compiler/llvm-ar",
        ),
        tool_path(
            name = "cpp",
            path = "/opt/intel/oneapi/compiler/2025.1/bin/compiler/clang-cpp",
        ),
        tool_path(
            name = "dwp",
            path = "/opt/intel/oneapi/compiler/2025.1/bin/compiler/llvm-dwp",
        ),
        tool_path(
            name = "llvm-cov",
            path = "/opt/intel/oneapi/compiler/2025.1/bin/compiler/llvm-cov",
        ),
        tool_path(
            name = "llvm-cov",
            path = "/opt/intel/oneapi/compiler/2025.1/bin/compiler/llvm-profdata",
        ),
        tool_path(
            name = "nm",
            path = "/opt/intel/oneapi/compiler/2025.1/bin/compiler/llvm-nm",
        ),
        tool_path(
            name = "objdump",
            path = "/usr/bin/objdump",
        ),
        tool_path(
            name = "strip",
            path = "/usr/bin/strip",
        ),
    ]

    features = [
        feature(
            name = "c_compiler_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-xc",
                                "-fPIC",
                            ]
                        )
                    ]
                )
            ]
        ),
        feature(
            name = "default_compiler_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-fPIC",
                                "-g0",
                                "-O2",
                                "-DNDEBUG",
                            ],
                        ),
                    ],
                ),
            ],
        ),
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-lstdc++",
                                "-lm",
                                "-fPIC",
                                "-fuse-ld=lld",
                                "-Wl,--build-id=md5",
                                "-Wl,--hash-style=gnu",
                                "-Wl,-z,relro,-z,now",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        cxx_builtin_include_directories = [
            "/opt/intel/oneapi/umf/0.10/include",
            "/opt/intel/oneapi/tbb/2022.1/env/../include",
            "/opt/intel/oneapi/pti/0.11/include",
            "/opt/intel/oneapi/mpi/2021.15/include",
            "/opt/intel/oneapi/mkl/2025.1/include",
            "/opt/intel/oneapi/ippcp/2025.1/include",
            "/opt/intel/oneapi/ipp/2022.1/include",
            "/opt/intel/oneapi/dpl/2022.8/include",
            "/opt/intel/oneapi/dpcpp-ct/2025.1/include",
            "/opt/intel/oneapi/dnnl/2025.1/include",
            "/opt/intel/oneapi/dev-utilities/2025.1/include",
            "/opt/intel/oneapi/dal/2025.4/include",
            "/opt/intel/oneapi/dal/2025.4/include/dal",
            "/opt/intel/oneapi/ccl/2021.15/include",
            "/opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/stl_wrappers",
            "/opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include",
            "/opt/intel/oneapi/compiler/2025.1/bin/compiler/../../opt/compiler/include",
            "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12",
            "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12",
            "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward",
            "/opt/intel/oneapi/compiler/2025.1/lib/clang/20/include",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu",
            "/usr/include",
        ],
        toolchain_identifier = "local",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "k8",
        target_libc = "unknown",
        compiler = "clang",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
