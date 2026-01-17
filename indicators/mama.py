import pandas as pd
import numpy as np
import math

class MAMA:
    """
    MESA Adaptive Moving Average (MAMA)
    
    Ported from Pine Script provided:
    f_mama(src, fast_limit, slow_limit)
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, fast_limit: float = 0.5, slow_limit: float = 0.05, source_col: str = 'close') -> pd.DataFrame:
        src = pd.to_numeric(df[source_col]).fillna(0).values
        n = len(src)
        
        # Output arrays
        mama = np.zeros(n)
        fama = np.zeros(n)
        
        # State variables
        period = 0.0
        smooth = 0.0
        detrender = 0.0
        i1 = 0.0
        q1 = 0.0
        jI = 0.0
        jQ = 0.0
        i2 = 0.0
        q2 = 0.0
        re = 0.0
        im = 0.0
        
        # Previous state tracking (arrays to simulate [1], [2] etc)
        # We need history for:
        # src: [1], [2], [3]
        # period: [1]
        # smooth: [2], [4], [6]
        # detrender: [2], [4], [6]
        # i1, q1: [2], [4], [6]
        # i2, q2: [1]
        # phase: [1]
        # mama: [1]
        # fama: [1]
        
        # To avoid massive manual distinct variables, we can use arrays and index i
        # But we need to update them iteratively.
        
        # Let's keep full arrays for intermediate variables to make 'nz(x[n])' easy
        vec_period = np.zeros(n)
        vec_smooth = np.zeros(n)
        vec_detrender = np.zeros(n)
        vec_i1 = np.zeros(n)
        vec_q1 = np.zeros(n)
        vec_i2 = np.zeros(n)
        vec_q2 = np.zeros(n)
        vec_phase = np.zeros(n)
        
        for i in range(n):
            if i < 6:
                # Startup phase, not enough data for 6 bars back
                mama[i] = src[i]
                fama[i] = src[i]
                continue
                
            # Helper to get lag
            def nz_val(arr, idx):
                if idx < 0: return 0.0
                return arr[idx]

            s0 = src[i]
            s1 = src[i-1]
            s2 = src[i-2]
            s3 = src[i-3]
            
            # Smooth
            # smooth := (4*src + 3*nz(src[1]) + 2*nz(src[2]) + nz(src[3])) / 10
            smooth = (4*s0 + 3*s1 + 2*s2 + s3) / 10.0
            vec_smooth[i] = smooth
            
            p1 = vec_period[i-1] # nz(period[1])
            
            # Detrender
            # detrender := (0.0962*smooth + 0.5769*nz(smooth[2]) - 0.5769*nz(smooth[4]) - 0.0962*nz(smooth[6])) * (0.075*nz(period[1]) + 0.54)
            sm0 = smooth
            sm2 = nz_val(vec_smooth, i-2)
            sm4 = nz_val(vec_smooth, i-4)
            sm6 = nz_val(vec_smooth, i-6)
            
            detrender = (0.0962*sm0 + 0.5769*sm2 - 0.5769*sm4 - 0.0962*sm6) * (0.075*p1 + 0.54)
            vec_detrender[i] = detrender
            
            # InPhase / Quadrature
            # q1 := (0.0962*detrender + 0.5769*nz(detrender[2]) - 0.5769*nz(detrender[4]) - 0.0962*nz(detrender[6])) * (0.075*nz(period[1]) + 0.54)
            d0 = detrender
            d2 = nz_val(vec_detrender, i-2)
            d4 = nz_val(vec_detrender, i-4)
            d6 = nz_val(vec_detrender, i-6)
            
            q1 = (0.0962*d0 + 0.5769*d2 - 0.5769*d4 - 0.0962*d6) * (0.075*p1 + 0.54)
            vec_q1[i] = q1
            
            # i1 := nz(detrender[3])
            i1 = nz_val(vec_detrender, i-3)
            vec_i1[i] = i1
            
            # Advance Phase
            # jI := (0.0962*i1 + 0.5769*nz(i1[2]) - 0.5769*nz(i1[4]) - 0.0962*nz(i1[6])) * (0.075*nz(period[1]) + 0.54)
            # jQ := (0.0962*q1 + 0.5769*nz(q1[2]) - 0.5769*nz(q1[4]) - 0.0962*nz(q1[6])) * (0.075*nz(period[1]) + 0.54)
            
            i1_0 = i1
            i1_2 = nz_val(vec_i1, i-2)
            i1_4 = nz_val(vec_i1, i-4)
            i1_6 = nz_val(vec_i1, i-6)
            
            jI = (0.0962*i1_0 + 0.5769*i1_2 - 0.5769*i1_4 - 0.0962*i1_6) * (0.075*p1 + 0.54)
            
            q1_0 = q1
            q1_2 = nz_val(vec_q1, i-2)
            q1_4 = nz_val(vec_q1, i-4)
            q1_6 = nz_val(vec_q1, i-6)
            
            jQ = (0.0962*q1_0 + 0.5769*q1_2 - 0.5769*q1_4 - 0.0962*q1_6) * (0.075*p1 + 0.54)
            
            # Phasor
            i2 = i1 - jQ
            q2 = q1 + jI
            
            # Smooth Period
            # i2 := 0.2*i2 + 0.8*nz(i2[1])
            # q2 := 0.2*q2 + 0.8*nz(q2[1])
            i2 = 0.2*i2 + 0.8*nz_val(vec_i2, i-1)
            q2 = 0.2*q2 + 0.8*nz_val(vec_q2, i-1)
            
            vec_i2[i] = i2
            vec_q2[i] = q2
            
            # Homodyne Discriminator
            # float re = i2*nz(i2[1]) + q2*nz(q2[1])
            # float im = i2*nz(q2[1]) - q2*nz(i2[1])
            re = i2*nz_val(vec_i2, i-1) + q2*nz_val(vec_q2, i-1)
            im = i2*nz_val(vec_q2, i-1) - q2*nz_val(vec_i2, i-1)
            
            # Smooth Period
            dp = 0.0
            if re != 0:
                dp = 2 * 3.14159 / math.atan(im/re)
            
            # Raw Limits
            # if dp > 1.5*nz(period[1]) dp := 1.5*nz(period[1])
            # if dp < 0.67*nz(period[1]) dp := 0.67*nz(period[1])
            if dp > 1.5 * p1: dp = 1.5 * p1
            if dp < 0.67 * p1: dp = 0.67 * p1
            if dp < 6: dp = 6
            if dp > 50: dp = 50
            
            period = 0.2*dp + 0.8*p1
            vec_period[i] = period
            
            # Phase
            phase = 0.0
            if i1 != 0:
                phase = math.atan(q1 / i1) * 180 / 3.14159
            
            # Delta Phase
            phase_prev = nz_val(vec_phase, i-1)
            delta_phase = phase_prev - phase
            if delta_phase < 1:
                delta_phase = 1
            
            vec_phase[i] = phase
            
            # Alpha
            alpha = fast_limit / delta_phase
            if alpha < slow_limit:
                alpha = slow_limit
            
            # MAMA
            # mama := alpha*src + (1 - alpha)*nz(mama[1])
            mama[i] = alpha*s0 + (1 - alpha)*mama[i-1]
            
            # FAMA
            # fama := 0.5*alpha*mama + (1 - 0.5*alpha)*nz(fama[1])
            fama[i] = 0.5*alpha*mama[i] + (1 - 0.5*alpha)*fama[i-1]
            
        return pd.DataFrame({'mama': mama, 'fama': fama}, index=df.index)
