#define TILE_DIM 32
#define BLOCK_ROWS 8

template <typename DTYPE>
__global__
void transpose4(
  const DTYPE* a,
  const DTYPE* b,
  const DTYPE* c,
  const DTYPE* d,
  DTYPE* a_t,
  DTYPE* b_t,
  DTYPE* c_t,
  DTYPE* d_t,
  int xdim,
  int ydim,
  int total_size
)
{
  __shared__ DTYPE tile[4*TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  
  if (x < xdim)
  {
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int index = (y+j)*xdim + x;
        if (index < total_size)
        {  
          tile[threadIdx.y+j][threadIdx.x] = a[index];
          tile[TILE_DIM + threadIdx.y+j][threadIdx.x] = b[index];
          tile[2 * TILE_DIM + threadIdx.y+j][threadIdx.x] = c[index];
          tile[3 * TILE_DIM + threadIdx.y+j][threadIdx.x] = d[index];
        }
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (x < ydim)
  {
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      int index = (y+j)*ydim + x;
      if (index < total_size)
      {
        
        a_t[index] = tile[threadIdx.x][threadIdx.y + j];
        b_t[index] = tile[TILE_DIM + threadIdx.x][threadIdx.y + j];
        c_t[index] = tile[2 * TILE_DIM + threadIdx.x][threadIdx.y + j];
        d_t[index] = tile[3 * TILE_DIM + threadIdx.x][threadIdx.y + j];
      }
    }
  }
}

template <typename DTYPE>
__global__
void transpose(
  const DTYPE* m,
  DTYPE* m_t,
  int xdim,
  int ydim,
  int total_size
)
{
  __shared__ DTYPE tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  
  if (x < xdim)
  {
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int index = (y+j)*xdim + x;
        if (index < total_size)
          tile[threadIdx.y+j][threadIdx.x] = m[index];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (x < ydim)
  {
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      int index = (y+j)*ydim + x;
      if (index < total_size)
        m_t[index] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}



template <typename DTYPE>
void transposeMats(const DTYPE* a, const DTYPE* b, const DTYPE* c, const DTYPE* d,
    DTYPE* a_t, DTYPE* b_t, DTYPE* c_t, DTYPE* d_t,
     int xdim, int ydim, int total_size)
{
    int num_b_x = (xdim + TILE_DIM -1) / TILE_DIM;
    int num_b_y = (ydim + TILE_DIM -1) / TILE_DIM;

    dim3 dimGrid(num_b_x, num_b_y, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    transpose4<<<dimGrid, dimBlock>>>(a, b, c, d, a_t, b_t, c_t, d_t, xdim, ydim, total_size);
}
template <typename DTYPE>
void transposeMat(const DTYPE* m, DTYPE* m_t, int xdim, int ydim, int total_size)
{
    int num_b_x = (xdim + TILE_DIM -1) / TILE_DIM;
    int num_b_y = (ydim + TILE_DIM -1) / TILE_DIM;

    dim3 dimGrid(num_b_x, num_b_y, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    transpose<<<dimGrid, dimBlock>>>(m, m_t, xdim, ydim, total_size);
}