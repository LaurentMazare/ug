import ug
from ug.lang import Arg, IndexExpr as I, Expr as E, Ops as O, Kernel

BLOCK_SIZE = 32

lhs_ptr = Arg.ptr_f32()
rhs_ptr = Arg.ptr_f32()
dst_ptr = Arg.ptr_f32()

offset = I.program_id() * I.cst(BLOCK_SIZE)
stride = I.cst(1)
len_ = I.cst(BLOCK_SIZE)
lhs = E.load(lhs_ptr, offset, len_, stride)
rhs = E.load(rhs_ptr, offset, len_, stride)
op = O.store(dst_ptr, offset, len_, stride, lhs + rhs)
kernel = Kernel("simple_add", [dst_ptr, lhs_ptr, rhs_ptr], [op])
print("==== LANG ====")
print(kernel)

ssa = kernel.lower()
print("==== SSA  ====")
print(ssa)
