import pyopencl as cl


BLOCK_SIDE = 8
BLOCKS_PER_DIM = 32

N = BLOCK_SIDE * BLOCKS_PER_DIM + 2

kernel = f"""
	inline int idx(int a, int b) {{
		return a*{N} + b;
	}}

	__kernel void gameStep(
		__global int *m,
		__global int *n,
		__global int *flag)
	{{
		int x = get_global_id(0);
		int y = get_global_id(1);

		int a00 = m[idx(x-1, y-1)];
		int a01 = m[idx(x-1, y  )];
		int a02 = m[idx(x-1, y+1)];
		
		int a10 = m[idx(x  , y-1)];
		int a11 = m[idx(x  , y  )];
		int a12 = m[idx(x  , y+1)];
		
		int a20 = m[idx(x+1, y-1)];
		int a21 = m[idx(x+1, y  )];
		int a22 = m[idx(x+1, y+1)];

		int s = a00 + a01 + a02
			  + a10       + a12
			  + a20 + a21 + a22;

		if (s < 2) {{
			n[idx(x, y)] = 0;
		}} else if (s == 2) {{
			n[idx(x, y)] = a11;
			
			if (a11) {{
				flag[idx(0, 0)] = 1;
			}}
		}} else if (s == 3) {{
			n[idx(x, y)] = 1;
			flag[idx(0, 0)] = 1;
		}} else if (s > 3) {{
			n[idx(x, y)] = 0;
		}}
	}}
"""

def main():
	context = cl.create_some_context()
	queue = cl.CommandQueue(context)
	program = cl.Program(context, kernel).build()
	device = queue.get_info(cl.command_queue_info.DEVICE)
	wg = program.gameStep.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)

	print(wg)
	
if __name__ == '__main__':
	main()