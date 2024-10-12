import ug

device = ug.Device(0)
instrs = [
    ug.Instr.define_global(0, ug.DType.ptr_f32),
    ug.Instr.define_global(1, ug.DType.ptr_f32),
    ug.Instr.define_global(2, ug.DType.ptr_f32),
]
kernel = ug.Kernel(instrs)
print(kernel.cuda_code())

s1 = device.slice([1.0, 2.0, 3.0, 4.0] * 32)
s2 = device.slice(range(128))
s_dst = device.zeros(128)
#func.launch3(s1, s2, s_dst)
#print(s_dst.to_vec())
