import ug
from ug import IndexExpr as I, Expr as E, Ops as O, Kernel

BLOCK_SIZE = 32

lhs_ptr = ug.Arg.ptr()
rhs_ptr = ug.Arg.ptr()
dst_ptr = ug.Arg.ptr()

offset = I.program_id() * I.cst(BLOCK_SIZE)
stride = I.cst(1)
len_ = I.cst(BLOCK_SIZE)
lhs = E.load(lhs_ptr, offset, len_, stride)
rhs = E.load(rhs_ptr, offset, len_, stride)
op = O.store(dst_ptr, offset, len_, stride, lhs + rhs)
kernel = Kernel("simple_add", [dst_ptr, lhs_ptr, rhs_ptr], [op])
print(kernel)
