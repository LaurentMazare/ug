import ug

device = ug.Device(0)
vec_len = 128
instrs = [
    ug.Instr.define_global(0, ug.DType.ptr_f32),
    ug.Instr.define_global(1, ug.DType.ptr_f32),
    ug.Instr.define_global(2, ug.DType.ptr_f32),
    ug.Instr.const_i32(0),
    ug.Instr.const_i32(vec_len),
    ug.Instr.range(3, 4, 11),
    ug.Instr.load(1, 5, ug.DType.f32),
    ug.Instr.load(2, 5, ug.DType.f32),
    ug.Instr.binary("add", 6, 7, ug.DType.f32),
    ug.Instr.store(0, 5, 8, ug.DType.f32),
    ug.Instr.end_range(5),
]
kernel = ug.Kernel(instrs)
cuda_code = kernel.cuda_code("sum")
print(cuda_code)

func = device.compile_cu(cuda_code, "sum")
s1 = device.slice([1.0, 2.0, 3.0, 4.0] * 32)
s2 = device.slice(range(128))
s_dst = device.zeros(128)

func.launch3(s_dst, s1, s2)
print(s_dst.to_vec())
