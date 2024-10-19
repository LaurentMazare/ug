import ug

def softmax(dim1, dim2):
    src_ptr = ug.op.Arg.ptr_f32()
    dst_ptr = ug.op.Arg.ptr_f32()

    src = ug.op.load(src_ptr, (dim1, dim2))
    exps = (src - src.max(1).broadcast(1, dim2)).exp()
    sm = exps / exps.sum(1).broadcast(1, dim2)
    st = ug.op.Store(dst_ptr, (dim1, dim2), sm)
    return ug.op.Kernel("softmax", [dst_ptr, src_ptr], [st])


sm = softmax(512, 128)
print(">>> KERNEL")
print(sm)
sm = sm.lower()
print(">>> SSA")
print(sm)
