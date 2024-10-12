import ug
import time

ptx_code = """
.version 6.0
.target sm_30
.address_size 64

.entry addSlices(
    .param .u64 par0,
    .param .u64 par1,
    .param .u64 par2
)
{
    .reg .u64 ptr0, ptr1, ptr2, tids;
    .reg .f32 r1, r2, r3;
    .reg .u32 tid;

    mov.u32 tid, %tid.x;

    ld.param.u64 ptr0, [par0];
    ld.param.u64 ptr1, [par1];
    ld.param.u64 ptr2, [par2];

    mul.wide.u32 tids, tid, 4;
    add.u64 ptr0, ptr0, tids;
    add.u64 ptr1, ptr1, tids;
    add.u64 ptr2, ptr2, tids;

    ld.global.f32 r1, [ptr0];
    ld.global.f32 r2, [ptr1];

    add.f32 r3, r1, r2;

    st.global.f32 [ptr2], r3;

    ret;
}
"""

device = ug.Device(0)
func = device.compile_ptx(ptx_code, "addSlices")

s1 = device.slice([1.0, 2.0, 3.0, 4.0] * 32)
s2 = device.slice(range(128))
s_dst = device.zeros(128)
func.launch3(s1, s2, s_dst)
print(s_dst.to_vec())
