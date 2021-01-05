type DTYPE = f32

let tridagSeq [m] (a:  [m]DTYPE, b: [m]DTYPE, c: [m]DTYPE, y: [m]DTYPE ): *[m]DTYPE =
   let cp = map (\i -> if i==0 then c[0]/b[0] else 0) (indices c)
   let yp = map (\i -> if i==0 then y[0]/b[0] else 0) (indices y)
   let (cp_full, yp_full) =
      loop (cp, yp) for i in 1..<m do
         let norm_factor = 1.0 / (b[i] - a[i] * cp[i-1])
         let cp[i] = c[i] * norm_factor
         let yp[i] = (y[i] - a[i] * yp[i-1]) * norm_factor
         in (cp, yp)
   let solution = replicate m (0.0 : DTYPE)
   let solution[m-1] = yp_full[m-1]
   let inds = reverse (init (indices y))
   in loop (solution) for i in inds do
      let solution[i] = yp_full[i] - cp_full[i] * solution[i+1]
      in solution

let tridiag [xdim][ydim][zdim] (a: [xdim][ydim][zdim]DTYPE) (b: [xdim][ydim][zdim]DTYPE) (c: [xdim][ydim][zdim]DTYPE) (y: [xdim][ydim][zdim]DTYPE): *[xdim][ydim][zdim]DTYPE =
   map4 (\as bs cs ys ->
        map4 (\a b c y ->
            tridagSeq (a,b,c,y)
        ) as bs cs ys
   ) a b c y

let limiter (cr: DTYPE) : DTYPE =
    f32.max 0 (f32.max (f32.min 1 (2 * cr)) (f32.min 2 cr))

let calcflux
    (velfac: DTYPE)
    (velS: DTYPE)
    (dt_tracer: DTYPE)
    (dx: DTYPE)
    (varS: DTYPE)
    (varSM1: DTYPE)
    (varSP1: DTYPE)
    (varSP2: DTYPE)
    (maskS: DTYPE)
    (maskSM1: DTYPE)
    (maskSP1: DTYPE)
    : DTYPE
  =
    let scaledVel =  velfac * velS
    let uCFL = f32.abs (scaledVel * dt_tracer / dx)
    let rjp = (varSP2 - varSP1) * maskSP1
    let rj = (varSP1 - varS) * maskS
    let rjm = (varS - varSM1) * maskSM1
    let epsilon = 0.00000000000000000001
    let divisor = if (f32.abs rj) < epsilon then epsilon else rj
    let cr = if velS>0 then rjm / divisor else rjp / divisor
    let cr = limiter(cr)
    in scaledVel * (varSP1 + varS) * 0.5 - (f32.abs scaledVel) * ((1.0 - cr) + uCFL * cr) * rj * 0.5


-- ==
-- entry: integrate_tke
--
-- compiled random input { [200][200][100]f32 [200][200][100]f32 [200][200][100]f32 [200][200][100]f32 [200][200][100]f32 [200][200][100]f32 
--                         [200][200][100]f32 [200][200][100]f32 [200][200][100]f32 [200][200][100]f32 [200][200][100]f32 [200][200][100]f32
--                         [200]f32 [200]f32 [200]f32 [200]f32 [100]f32 [100]f32 [200]f32 [200]f32 [200][200]i32
--                         [200][200][100]f32 [200][200][100]f32 [200][200][100]f32 [200][200]f32}
entry integrate_tke [xdim][ydim][zdim]
        (tketau: * [xdim][ydim][zdim]DTYPE)
        (tketaup1:*[xdim][ydim][zdim]DTYPE)
        (tketaum1:*[xdim][ydim][zdim]DTYPE)
        (dtketau: *[xdim][ydim][zdim]DTYPE)
       (dtketaup1:*[xdim][ydim][zdim]DTYPE)
       (dtketaum1:*[xdim][ydim][zdim]DTYPE)
        (utau:     [xdim][ydim][zdim]DTYPE)
        -- (utaup1:   [xdim][ydim][zdim]DTYPE)
        -- (utaum1:   [xdim][ydim][zdim]DTYPE)
        (vtau:     [xdim][ydim][zdim]DTYPE)
        -- (vtaup1:   [xdim][ydim][zdim]DTYPE)
        -- (vtaum1:   [xdim][ydim][zdim]DTYPE)
        (wtau:     [xdim][ydim][zdim]DTYPE)
        -- (wtaup1:   [xdim][ydim][zdim]DTYPE)
        -- (wtaum1:   [xdim][ydim][zdim]DTYPE)
        (maskU:    [xdim][ydim][zdim]DTYPE)
        (maskV:    [xdim][ydim][zdim]DTYPE)
        (maskW:    [xdim][ydim][zdim]DTYPE)
        (dxt:      [xdim]            DTYPE)
        (dxu:      [xdim]            DTYPE)
        (dyt:            [ydim]      DTYPE)
        (dyu:            [ydim]      DTYPE)
        (dzt:                  [zdim]DTYPE)
        (dzw:                  [zdim]DTYPE)
        (cost:           [ydim]      DTYPE)
        (cosu:           [ydim]      DTYPE)
        (kbot:     [xdim][ydim]      i32  )
        (kappaM:   [xdim][ydim][zdim]DTYPE)
        (mxl:      [xdim][ydim][zdim]DTYPE)
        (forc:     [xdim][ydim][zdim]DTYPE)
        (forc_tke_surface:
                   [xdim][ydim]    DTYPE) =
        --  :
            -- (tke: *[3][h][w][d]DTYPE,
            -- dtke: *[3][h][w][d]DTYPE,
            -- tke_surf_corr: [h][w]DTYPE) =
    -- let tau = 0
    -- let taup1 = 1
    -- let taum1 = 2
    let dt_tracer = 1
    let dt_mom = 1
    let AB_eps = 0.1
    let alpha_tke = 1
    let c_eps = 0.7
    let K_h_tke = 2000
    let dt_tke = dt_mom

    -- -- Init delta
    let tridiags = tabulate_3d xdim ydim zdim (\x y z ->
                    if x >= 2 && x < xdim - 2 && y >= 2 && y < ydim - 2
                        then
                            let tke = tketau[x,y,z]
                            let sqrttke = f32.sqrt (f32.max 0 tke)
                            let ks_val = kbot[x,y]-1
                            let land_mask = ks_val >= 0
                            let edge_mask = land_mask && ((i32.i64 z) == ks_val)
                            let water_mask = land_mask && ((i32.i64 z) >= ks_val)
                            
                            let kappa = kappaM[x, y, z]
                            let deltam1 = if z > 0 then dt_tke / dzt[z] * alpha_tke * 0.5
                                * (kappaM[x, y, z-1] + kappa) else 0
                            let delta = if z < zdim-1 then dt_tke / dzt[z+1] * alpha_tke * 0.5
                                * (kappa + kappaM[x, y, z+1]) else 0
                            let dzwz = dzw[z]
                            let mxls = mxl[x, y, z]
                            let a = 
                                if edge_mask || (!water_mask)
                                    then 0
                                else
                                    if z > 0 && z < zdim-1
                                        then -deltam1 / dzwz
                                    else if z == zdim-1
                                        then -deltam1 / (0.5 * dzwz)
                                    else 0
                            let b = if !water_mask
                                        then 1
                                    else if edge_mask
                                    then
                                        1 + delta / dzwz
                                            + dt_tke * c_eps / mxls * sqrttke
                                    else
                                        if z > 0 && z < zdim-1
                                            then 1 + (delta + deltam1) / dzwz
                                                + dt_tke * c_eps
                                                * sqrttke / mxls
                                        else if z == zdim-1
                                            then 1 + deltam1 / (0.5 * dzwz)
                                            + dt_tke * c_eps / mxls * sqrttke
                                        else 0
                            let (c,d) = 
                                if !water_mask
                                then
                                    (0,0)
                                else
                                    let c = -delta / dzwz 
                                    let tmp = tke + dt_tke * forc[x,y,z]
                                    in if z == zdim-1
                                        then (c, tmp + dt_tke * forc_tke_surface[x,y] / (0.5 * dzwz))
                                        else (c, tmp)
                            in (a,b,c,d)
                        else (0,0,0,0)
                )

    let (a, b, c, d) = unzip4 (map (\xs ->
                            unzip4 (map (\ys ->
                                unzip4 ys
                            ) xs
                        )
                       ) tridiags)
    let sol = tridiag a b c d

    -- lateral diff east
    let flux_east_latdiff = tabulate_3d xdim ydim zdim (\x y z ->
                        if x < xdim-1
                        then
                            K_h_tke * (tketau[x+1, y, z] - tketau[x, y, z])
                                / (cost[y] * dxu[x]) * maskU[x, y, z]
                        else 0

    )
    -- lateral diff north
    let flux_north_latdiff = tabulate_3d xdim ydim zdim (\x y z ->
                        if y < ydim-1
                        then
                            K_h_tke * (tketau[x, y+1, z] - tketau[x, y, z])
                                / dyu[y] * maskV[x, y, z] * cosu[y]
                        else 0
                    )
    

    let tketaup1 = tabulate_3d xdim ydim zdim (\x y z ->
                let ks_val = kbot[x,y]-1
                let water_mask = (ks_val >= 0) && ((i32.i64 z) >= ks_val) in
                    if x >= 2 && x < xdim - 2 && y >= 2 && y < ydim - 2 && water_mask
                        then sol[x,y,z]
                        else tketaup1[x,y,z]
                )
    let tke_surf_corr = tabulate_2d xdim ydim (\x y ->
                    if x >= 2 && x < xdim - 2 && y >= 2 && y < ydim - 2
                    then
                        let tke_val = tketaup1[x, y, zdim-1] in
                            if tke_val < 0
                            then -tke_val * 0.5 * dzw[zdim-1] / dt_tke
                            else 0
                    else 0
                    )
    
    
    -- clip negative vals on last z val



    -- -- tendency due to advection
    let flux_east = tabulate_3d xdim ydim zdim (\x y z ->
                        if x >= 1 && x < xdim - 2 && y >= 2 && y < ydim - 2
                        then
                            let dx = cost[y] * dxt[x]
                            let velS = utau[x,y,z]

                            let maskWm1 = maskW[x-1,y,z]
                            let maskWs = maskW[x,y,z]
                            let maskWp1 = maskW[x+1,y,z]
                            let maskwp2 = maskW[x+2,y,z]
                            let varSM1 = tketau[x-1,y,z]
                            let varS = tketau[x,y,z]
                            let varSP1 = tketau[x+1,y,z]
                            let varSP2 = tketau[x+2,y,z]
                            let maskUtr = if x < xdim-1 then maskWs * maskWp1 else 0
                            let maskUtrP1 = if x < xdim-1 then maskWp1 * maskwp2 else 0
                            let maskUtrM1 = if x < xdim-1 then maskWm1 * maskWs else 0
                            in calcflux 1 velS dt_tracer dx varS varSM1 varSP1 varSP2 maskUtr maskUtrM1 maskUtrP1
                        else 0
                    )
    let flux_north = tabulate_3d xdim ydim zdim (\x y z ->
                        if y >= 1 && y < ydim - 2 && x >= 2 && x < xdim-2
                        then
                            let dx = cost[y] * dyt[y]
                            let velS = vtau[x,y,z]
                            let maskWm1 = maskW[x,y-1,z]
                            let maskWs = maskW[x,y,z]
                            let maskWp1 = maskW[x,y+1,z]
                            let maskwp2 = maskW[x,y+2,z]
                            let varSM1 = tketau[x,y-1,z]
                            let varS = tketau[x,y,z]
                            let varSP1 = tketau[x,y+1,z]
                            let varSP2 = tketau[x,y+2,z]
                            let maskVtr = if y < ydim-1 then maskWs * maskWp1 else 0
                            let maskVtrP1 = if y < ydim-1 then maskWp1 * maskwp2 else 0
                            let maskVtrM1 = if y < ydim-1 then maskWm1 * maskWs else 0
                            in calcflux cosu[y] velS dt_tracer dx varS varSM1 varSP1 varSP2 maskVtr maskVtrM1 maskVtrP1
                        else 0
                    )
    let flux_top = tabulate_3d xdim ydim zdim (\x y z ->
                        if z < zdim-1 && x >= 2 && x < xdim-2 && y >= 2 && y < ydim-2
                        then
                            let velS = wtau[x,y,z]
                            let varSM1 = if z != 0 then tketau[x,y,z-1] else 0
                            let varS = tketau[x,y,z]
                            let varSP2 = if z < zdim-2 then tketau[x,y,z+2] else 0
                            let varSP1 = tketau[x,y,z+1]
                            let maskWm1 = if z != 0 then maskW[x,y,z-1] else 0
                            let maskWs = maskW[x,y,z]
                            let maskWp1 = maskW[x,y,z+1]
                            let maskwp2 = if z < zdim-2 then maskW[x,y,z+2] else 0
                            let maskWtr = maskWs * maskWp1
                            let maskWtrP1 = maskWp1 * maskwp2
                            let maskWtrM1 = maskWm1 * maskWs
                            let dx = dzw[z]
                            in calcflux 1 velS dt_tracer dx varS varSM1 varSP1 varSP2 maskWtr maskWtrM1 maskWtrP1
                        else 0
                    )


    let tketaup1dtketau = tabulate_3d xdim ydim zdim (\x y z ->
                        let costy = cost[y]
                        let flux_east_factor = (costy * dxt[x])
                        let flux_north_factor = (costy * dyt[y])
                        let dtke_init = 
                            if x >= 2 && x < xdim - 2 && y >= 2 && y < ydim - 2
                            then
                                maskW[x,y,z] * (-(flux_east[x,y,z] - flux_east[x-1, y, z])
                                    / flux_east_factor
                                    - (flux_north[x,y,z] - flux_north[x, y-1, z])
                                    / flux_north_factor)
                            else dtketau[x,y,z]

                        let z0_update = if z == 0 then dtke_init - flux_top[x, y, 0] / dzw[0] else dtke_init
                        let z_middle_update = if z >= 1 && z < zdim-1 
                                                then z0_update - (flux_top[x, y, z] - flux_top[x, y, z-1]) / dzw[z]
                                                else z0_update
                        let dtketau =  if z == zdim-1 then z_middle_update - (flux_top[x, y, z] - flux_top[x, y, z-1]) /
                                                        (0.5 * dzw[z])
                                          else z_middle_update

                        let tke_val = tketaup1[x,y,z]
                        let step = dt_tracer * ((1.5 + AB_eps) * dtketau - (0.5 + AB_eps) * dtketaum1[x, y, z]) in
                        if x >= 2 && x < xdim - 2 && y >= 2 && y < ydim - 2 && z == zdim-1
                        then
                            
                            let tke_val_clipped = 
                                if tke_val < 0
                                    then 0
                                    else tke_val
                            in (tke_val_clipped + dt_tke * maskW[x, y, z] *
                                ((flux_east_latdiff[x,y,z] - flux_east_latdiff[x-1, y, z])
                                / flux_east_factor
                                + (flux_north_latdiff[x,y,z] - flux_north_latdiff[x, y-1, z])
                                / flux_north_factor) + step, dtketau)
                        else (tke_val + step, dtketau)
                    )
    let (tketaup1, dtketau) =  unzip2 (map (\xs ->
                                    unzip2 (map (\ys ->
                                        unzip2 ys
                                    ) xs)
                            ) tketaup1dtketau)

    in (tketau, tketaup1, tketaum1, dtketau, dtketaup1, dtketaum1, tke_surf_corr)