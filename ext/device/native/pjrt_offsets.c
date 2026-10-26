// The layout of the PJRT C API structs that gen.device.pjrt uses, printed
// as EDN by the C compiler, for that namespace's `layout`. Generated on the
// target (x86_64 Linux, gcc), from this directory:
//
//   gcc -std=c11 -Wall -o /tmp/pjrt_offsets pjrt_offsets.c && /tmp/pjrt_offsets
//
// Each struct gives :size (its *_STRUCT_SIZE, which the struct_size field
// carries), :sizeof (what to allocate), and for each field
// [offset size kind], kind :ptr, :int, :float or :struct (from
// __builtin_classify_type, so the compiler says which fields are pointers).

#include <stdio.h>

#include "pjrt_c_api.h"

static int first;

static const char* kind(int type_class) {
  switch (type_class) {
    case 5: return ":ptr";      // pointer_type_class
    case 8: return ":float";    // real_type_class
    case 12: return ":struct";  // record_type_class
    default: return ":int";     // integer, char, enumeral, boolean
  }
}

#define STRUCT(s, size)                                               \
  printf("\n  :%s\n  {:size %zu :sizeof %zu\n   :fields {", #s,       \
         (size_t)(size), sizeof(s));                                  \
  first = 1
#define ARGS(s) STRUCT(s, s##_STRUCT_SIZE)
#define F(s, f)                                                       \
  do {                                                                \
    printf("%s:%s [%zu %zu %s]", first ? "" : "\n            ", #f,  \
           offsetof(s, f), sizeof(((s*)0)->f),                        \
           kind(__builtin_classify_type(((s*)0)->f)));                \
    first = 0;                                                        \
  } while (0)
#define END() printf("}}")
#define ENUM(e) printf("\n  :%s %d", #e, (int)(e))

int main(void) {
  printf("{:header [%d %d]\n :structs\n {", PJRT_API_MAJOR, PJRT_API_MINOR);

  STRUCT(PJRT_Api, PJRT_Api_STRUCT_SIZE);
  F(PJRT_Api, pjrt_api_version);
  F(PJRT_Api, PJRT_Error_Destroy);
  F(PJRT_Api, PJRT_Error_Message);
  F(PJRT_Api, PJRT_Error_GetCode);
  F(PJRT_Api, PJRT_Plugin_Initialize);
  F(PJRT_Api, PJRT_Event_Destroy);
  F(PJRT_Api, PJRT_Event_Await);
  F(PJRT_Api, PJRT_Client_Create);
  F(PJRT_Api, PJRT_Client_Destroy);
  F(PJRT_Api, PJRT_Client_PlatformName);
  F(PJRT_Api, PJRT_Client_PlatformVersion);
  F(PJRT_Api, PJRT_Client_AddressableDevices);
  F(PJRT_Api, PJRT_Client_Compile);
  F(PJRT_Api, PJRT_Client_BufferFromHostBuffer);
  F(PJRT_Api, PJRT_DeviceDescription_Kind);
  F(PJRT_Api, PJRT_Device_GetDescription);
  F(PJRT_Api, PJRT_Executable_Destroy);
  F(PJRT_Api, PJRT_Executable_NumOutputs);
  F(PJRT_Api, PJRT_LoadedExecutable_Destroy);
  F(PJRT_Api, PJRT_LoadedExecutable_GetExecutable);
  F(PJRT_Api, PJRT_LoadedExecutable_Execute);
  F(PJRT_Api, PJRT_Buffer_Destroy);
  F(PJRT_Api, PJRT_Buffer_ElementType);
  F(PJRT_Api, PJRT_Buffer_Dimensions);
  F(PJRT_Api, PJRT_Buffer_ToHostBuffer);
  END();

  ARGS(PJRT_Api_Version);
  F(PJRT_Api_Version, major_version);
  F(PJRT_Api_Version, minor_version);
  END();

  ARGS(PJRT_Error_Destroy_Args);
  F(PJRT_Error_Destroy_Args, error);
  END();

  ARGS(PJRT_Error_Message_Args);
  F(PJRT_Error_Message_Args, error);
  F(PJRT_Error_Message_Args, message);
  F(PJRT_Error_Message_Args, message_size);
  END();

  ARGS(PJRT_Error_GetCode_Args);
  F(PJRT_Error_GetCode_Args, error);
  F(PJRT_Error_GetCode_Args, code);
  END();

  ARGS(PJRT_Plugin_Initialize_Args);
  END();

  ARGS(PJRT_Event_Destroy_Args);
  F(PJRT_Event_Destroy_Args, event);
  END();

  ARGS(PJRT_Event_Await_Args);
  F(PJRT_Event_Await_Args, event);
  END();

  ARGS(PJRT_NamedValue);
  F(PJRT_NamedValue, name);
  F(PJRT_NamedValue, name_size);
  F(PJRT_NamedValue, type);
  F(PJRT_NamedValue, string_value);
  F(PJRT_NamedValue, int64_value);
  F(PJRT_NamedValue, int64_array_value);
  F(PJRT_NamedValue, float_value);
  F(PJRT_NamedValue, bool_value);
  F(PJRT_NamedValue, value_size);
  END();

  ARGS(PJRT_Client_Create_Args);
  F(PJRT_Client_Create_Args, create_options);
  F(PJRT_Client_Create_Args, num_options);
  F(PJRT_Client_Create_Args, client);
  END();

  ARGS(PJRT_Client_Destroy_Args);
  F(PJRT_Client_Destroy_Args, client);
  END();

  ARGS(PJRT_Client_PlatformName_Args);
  F(PJRT_Client_PlatformName_Args, client);
  F(PJRT_Client_PlatformName_Args, platform_name);
  F(PJRT_Client_PlatformName_Args, platform_name_size);
  END();

  ARGS(PJRT_Client_PlatformVersion_Args);
  F(PJRT_Client_PlatformVersion_Args, client);
  F(PJRT_Client_PlatformVersion_Args, platform_version);
  F(PJRT_Client_PlatformVersion_Args, platform_version_size);
  END();

  ARGS(PJRT_Client_AddressableDevices_Args);
  F(PJRT_Client_AddressableDevices_Args, client);
  F(PJRT_Client_AddressableDevices_Args, addressable_devices);
  F(PJRT_Client_AddressableDevices_Args, num_addressable_devices);
  END();

  ARGS(PJRT_Device_GetDescription_Args);
  F(PJRT_Device_GetDescription_Args, device);
  F(PJRT_Device_GetDescription_Args, device_description);
  END();

  ARGS(PJRT_DeviceDescription_Kind_Args);
  F(PJRT_DeviceDescription_Kind_Args, device_description);
  F(PJRT_DeviceDescription_Kind_Args, device_kind);
  F(PJRT_DeviceDescription_Kind_Args, device_kind_size);
  END();

  ARGS(PJRT_Program);
  F(PJRT_Program, code);
  F(PJRT_Program, code_size);
  F(PJRT_Program, format);
  F(PJRT_Program, format_size);
  END();

  ARGS(PJRT_Client_Compile_Args);
  F(PJRT_Client_Compile_Args, client);
  F(PJRT_Client_Compile_Args, program);
  F(PJRT_Client_Compile_Args, compile_options);
  F(PJRT_Client_Compile_Args, compile_options_size);
  F(PJRT_Client_Compile_Args, executable);
  END();

  ARGS(PJRT_LoadedExecutable_Destroy_Args);
  F(PJRT_LoadedExecutable_Destroy_Args, executable);
  END();

  ARGS(PJRT_LoadedExecutable_GetExecutable_Args);
  F(PJRT_LoadedExecutable_GetExecutable_Args, loaded_executable);
  F(PJRT_LoadedExecutable_GetExecutable_Args, executable);
  END();

  ARGS(PJRT_Executable_Destroy_Args);
  F(PJRT_Executable_Destroy_Args, executable);
  END();

  ARGS(PJRT_Executable_NumOutputs_Args);
  F(PJRT_Executable_NumOutputs_Args, executable);
  F(PJRT_Executable_NumOutputs_Args, num_outputs);
  END();

  ARGS(PJRT_ExecuteOptions);
  F(PJRT_ExecuteOptions, launch_id);
  END();

  ARGS(PJRT_LoadedExecutable_Execute_Args);
  F(PJRT_LoadedExecutable_Execute_Args, executable);
  F(PJRT_LoadedExecutable_Execute_Args, options);
  F(PJRT_LoadedExecutable_Execute_Args, argument_lists);
  F(PJRT_LoadedExecutable_Execute_Args, num_devices);
  F(PJRT_LoadedExecutable_Execute_Args, num_args);
  F(PJRT_LoadedExecutable_Execute_Args, output_lists);
  F(PJRT_LoadedExecutable_Execute_Args, device_complete_events);
  F(PJRT_LoadedExecutable_Execute_Args, execute_device);
  END();

  ARGS(PJRT_Client_BufferFromHostBuffer_Args);
  F(PJRT_Client_BufferFromHostBuffer_Args, client);
  F(PJRT_Client_BufferFromHostBuffer_Args, data);
  F(PJRT_Client_BufferFromHostBuffer_Args, type);
  F(PJRT_Client_BufferFromHostBuffer_Args, dims);
  F(PJRT_Client_BufferFromHostBuffer_Args, num_dims);
  F(PJRT_Client_BufferFromHostBuffer_Args, byte_strides);
  F(PJRT_Client_BufferFromHostBuffer_Args, num_byte_strides);
  F(PJRT_Client_BufferFromHostBuffer_Args, host_buffer_semantics);
  F(PJRT_Client_BufferFromHostBuffer_Args, device);
  F(PJRT_Client_BufferFromHostBuffer_Args, memory);
  F(PJRT_Client_BufferFromHostBuffer_Args, device_layout);
  F(PJRT_Client_BufferFromHostBuffer_Args, done_with_host_buffer);
  F(PJRT_Client_BufferFromHostBuffer_Args, buffer);
  END();

  ARGS(PJRT_Buffer_Destroy_Args);
  F(PJRT_Buffer_Destroy_Args, buffer);
  END();

  ARGS(PJRT_Buffer_ElementType_Args);
  F(PJRT_Buffer_ElementType_Args, buffer);
  F(PJRT_Buffer_ElementType_Args, type);
  END();

  ARGS(PJRT_Buffer_Dimensions_Args);
  F(PJRT_Buffer_Dimensions_Args, buffer);
  F(PJRT_Buffer_Dimensions_Args, dims);
  F(PJRT_Buffer_Dimensions_Args, num_dims);
  END();

  ARGS(PJRT_Buffer_ToHostBuffer_Args);
  F(PJRT_Buffer_ToHostBuffer_Args, src);
  F(PJRT_Buffer_ToHostBuffer_Args, host_layout);
  F(PJRT_Buffer_ToHostBuffer_Args, dst);
  F(PJRT_Buffer_ToHostBuffer_Args, dst_size);
  F(PJRT_Buffer_ToHostBuffer_Args, event);
  END();
  printf("}\n :enums\n {");
  ENUM(PJRT_Buffer_Type_PRED);
  ENUM(PJRT_Buffer_Type_S32);
  ENUM(PJRT_Buffer_Type_U32);
  ENUM(PJRT_Buffer_Type_F32);
  ENUM(PJRT_Buffer_Type_F64);
  ENUM(PJRT_NamedValue_kString);
  ENUM(PJRT_NamedValue_kInt64);
  ENUM(PJRT_NamedValue_kInt64List);
  ENUM(PJRT_NamedValue_kFloat);
  ENUM(PJRT_NamedValue_kBool);
  ENUM(PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes);
  printf("}}\n");
  return 0;
}
