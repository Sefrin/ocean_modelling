-- segmented scan with (+) on floats:
let sgmSumf64 [n] (flg : [n]i32) (arr : [n]f64) : [n]f64 =
  let flgs_vals =
    scan ( \ (f1, x1) (f2,x2) ->
            let f = f1 | f2 in
            if f2 > 0 then (f, x2)
            else (f, x1 + x2) )
         (0,0.0f64) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals


let sgmScanRec1 [n] (flg: [n](i32)) (arr: [n](f64,f64,f64,f64)) : [n](f64,f64,f64,f64) =
   let flgs_vals =
      scan (\(f1, (a0,a1,a2,a3)) (f2,(b0,b1,b2,b3)) ->
               let f = f1 | f2 in
                  if f2 > 0 then (f, (b0,b1,b2,b3))
                  else 
                     let value = 1.0/(a0*b0)
                     let newTuple  =   ((b0*a0 + b1*a2)*value,
                                       (b0*a1 + b1*a3)*value,
                                       (b2*a0 + b3*a2)*value,
                                       (b2*a1 + b3*a3)*value)
                     in (f, newTuple))
            (0,(1.0f64, 0.0f64, 0.0f64, 1.0f64)) (zip flg arr)
   let (_, vals) = unzip flgs_vals
   in vals

let sgmScanRec23 [n] (flg: [n](i32)) (arr: [n](f64,f64)) : [n](f64,f64) =
   let flgs_vals =
   scan ( \ (f1, (a0,a1)) (f2,(b0,b1)) ->
         let f = f1 | f2 in
         if f2 > 0 then (f, (b0,b1))
         else 
            let newTuple = ( b0 + b1*a0, a1*b1 )
            in (f, newTuple))
      (0,(0.0f64, 1.0f64)) (zip flg arr)
   let (_, vals) = unzip flgs_vals
   in vals



let tridagPar [n] (a:  [n]f64, b: [n]f64, c: [n]f64, y: [n]f64 ): *[n]f64 =
  #[unsafe]
  ----------------------------------------------------
  -- Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
  --   solved by scan with 2x2 matrix mult operator --
  ----------------------------------------------------
  let b0   = b[0]
  let mats = map  (\(i: i32): (f64,f64,f64,f64)  ->
                     if 0 < i
                     then (b[i], 0.0-a[i]*c[i-1], 1.0, 0.0)
                     else (1.0,  0.0,             0.0, 1.0))
                  (iota n)
  let scmt = scan (\(a0,a1,a2,a3) (b0,b1,b2,b3) ->
                     let value = 1.0/(a0*b0)
                     in ( (b0*a0 + b1*a2)*value,
                          (b0*a1 + b1*a3)*value,
                          (b2*a0 + b3*a2)*value,
                          (b2*a1 + b3*a3)*value))
                  (1.0,  0.0, 0.0, 1.0) mats
  let b    = map (\(t0,t1,t2,t3) ->
                    (t0*b0 + t1) / (t2*b0 + t3))
                 scmt
  ------------------------------------------------------
  -- Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
  --   solved by scan with linear func comp operator  --
  ------------------------------------------------------
  let y0   = y[0]
  let lfuns= map  (\(i: i32): (f64,f64)  ->
                     if 0 < i
                     then (y[i], 0.0-a[i]/b[i-1])
                     else (0.0,  1.0))
                  (iota n)
  let cfuns = scan (\(a: (f64,f64)) (b: (f64,f64)): (f64,f64)  ->
                     let (a0,a1) = a
                     let (b0,b1) = b
                     in ( b0 + b1*a0, a1*b1 ))
                  (0.0, 1.0) lfuns
  let y    = map (\(tup: (f64,f64)): f64  ->
                    let (a,b) = tup
                    in a + b*y0)
                 cfuns
  ------------------------------------------------------
  -- Recurrence 3: backward recurrence solved via     --
  --             scan with linear func comp operator  --
  ------------------------------------------------------
  let yn   = y[n-1]/b[n-1]
  let lfuns= map (\(k: i32): (f64,f64)  ->
                    let i = n-k-1
                    in  if   0 < k
                        then (y[i]/b[i], 0.0-c[i]/b[i])
                        else (0.0,       1.0))
                 (iota n)
  let cfuns= scan (\(a: (f64,f64)) (b: (f64,f64)): (f64,f64)  ->
                     let (a0,a1) = a
                     let (b0,b1) = b
                     in (b0 + b1*a0, a1*b1))
                  (0.0, 1.0) lfuns
  let y    = map (\(tup: (f64,f64)): f64  ->
                    let (a,b) = tup
                    in a + b*yn)
                 cfuns
  let y    = map (\i -> y[n-i-1]) (iota n)
  in y

entry tridagNested [n][m] (a: [n][m]f64) (b: [n][m]f64) (c: [n][m]f64) (y: [n][m]f64): *[n][m]f64 =
   map4 (\a b c y -> tridagPar (a,b,c,y)) a b c y


entry tridagParFlat [n] (a:  [n]f64) (b: [n]f64) (c: [n]f64) (y: [n]f64 ) (segSize: i32) (segCount: i32): [n]f64 =
  #[unsafe]
  ----------------------------------------------------
  -- Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
  --   solved by scan with 2x2 matrix mult operator --
  ----------------------------------------------------
   let inds = map (\i -> i * segSize) (iota segCount)
   let flags = scatter (replicate n 0) inds (replicate segCount 1)

   let bstarts = map (\i -> b[i]) inds
   let b0_flag = scatter (replicate n 0) inds bstarts
   let b0s = sgmSumf64 (map (\f -> if f>0 then 1 else 0 ) b0_flag) b0_flag
   -- let b0   = b[0]
   let mats = map  (\(i: i32): (f64,f64,f64,f64)  ->
                        if 0 < (i % segSize)
                        then (b[i], 0.0-a[i]*c[i-1], 1.0, 0.0)
                        else (1.0f64,  0.0f64,             0.0f64, 1.0f64))
                     (iota n)
   -- let scmt = scan (\(a0,a1,a2,a3) (b0,b1,b2,b3) ->
   --                      let value = 1.0/(a0*b0)
   --                      in ( (b0*a0 + b1*a2)*value,
   --                         (b0*a1 + b1*a3)*value,
   --                         (b2*a0 + b3*a2)*value,
   --                         (b2*a1 + b3*a3)*value))
                     -- (1.0,  0.0, 0.0, 1.0) mats
   let scmt = sgmScanRec1 flags mats                     
   let b    = map2 (\(t0,t1,t2,t3) b0 ->
                     (t0*b0 + t1) / (t2*b0 + t3))
                  scmt b0s
   ------------------------------------------------------
   -- Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
   --   solved by scan with linear func comp operator  --
   ------------------------------------------------------
   -- let y0   = y[0]
   let ystarts = map (\i -> y[i]) inds
   let y0_flag = scatter (replicate n 0) inds ystarts
   let y0s = sgmSumf64 (map (\f -> if f>0 then 1 else 0 ) y0_flag) y0_flag
   let lfuns = map  (\(i: i32): (f64,f64)  ->
                        if 0 < (i % segSize)
                        then (y[i], 0.0-a[i]/b[i-1])
                        else (0.0,  1.0))
                     (iota n)
   -- let cfuns= scan (\(a: (f64,f64)) (b: (f64,f64)): (f64,f64)  ->
   --                      let (a0,a1) = a
   --                      let (b0,b1) = b
   --                      in ( b0 + b1*a0, a1*b1 ))
   --                   (0.0, 1.0) lfuns
   let cfuns = sgmScanRec23 flags lfuns
   let y    = map2 (\(tup: (f64,f64)) y0: f64  ->
                     let (a,b) = tup
                     in a + b*y0)
                  cfuns y0s
   ------------------------------------------------------
   -- Recurrence 3: backward recurrence solved via     --
   --             scan with linear func comp operator  --
   ------------------------------------------------------

   let yb_ends = map (\i -> y[i+segSize-1]/b[i+segSize-1]) inds
   let ybends_flag = scatter (replicate n 0) inds yb_ends
   let yns = sgmSumf64 (map (\f -> if f>0 then 1 else 0) ybends_flag)  ybends_flag
   -- let yn   = y[n-1]/b[n-1]
   let lfuns= map (\(k: i32): (f64,f64)  ->
                     let seg = k / segSize
                     let segInd = k % segSize
                     let i = segSize * seg + segSize-segInd-1
                     in  if   0 < i
                           then (y[i]/b[i], 0.0-c[i]/b[i])
                           else (0.0,       1.0))
                  (iota n)
   -- let cfuns = scan (\(a: (f64,f64)) (b: (f64,f64)): (f64,f64)  ->
   --                      let (a0,a1) = a
   --                      let (b0,b1) = b
   --                      in (b0 + b1*a0, a1*b1))
   --                   (0.0, 1.0) lfuns
   let cfuns = sgmScanRec23 flags lfuns
   let y    = map2 (\(tup: (f64,f64)) yn: f64  ->
                     let (a,b) = tup
                     in a + b*yn)
                  cfuns yns
   let y    = map (\k -> 
                     let seg = k / segSize
                     let segInd = k % segSize
                     let i = segSize * seg + segSize-segInd-1
                     in y[i]
                  ) (iota n)
   in y
   