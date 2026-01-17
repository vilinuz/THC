import pandas as pd
import numpy as np
import math

class SuperSmoother:
    """
    Ehlers SuperSmoother Filter
    
    Pine Script:
    f_supersmoother(src, len) =>
        float a1 = math.exp(-1.414 * 3.14159 / len)
        float b1 = 2 * a1 * math.cos(1.414 * 180 / len) // Degrees in Pine 180/len? 
        // Note: Pine math.cos takes radians. The source code had `1.414 * 180 / len` inside cos?
        // Wait, standard Ehlers formula uses radians: 1.414 * pi / len.
        // The user provided source says: `math.cos(1.414 * 180 / len)` accompanied by comment "// Cosine in Degrees? Pine uses radians".
        // And then immediately calculates `float arg = 1.414 * 3.14159 / len` and uses `math.cos(arg)`.
        // The script actually uses `arg` which is radians (using 3.14159).
        
        float arg = 1.414 * 3.14159 / len
        float c1 = 2 * a1 * math.cos(arg)
        float c2 = -a1 * a1
        float c3 = 1 - c1 - c2
        float ss = 0.0
        ss := c3 * (src + nz(src[1])) / 2 + c1 * nz(ss[1]) + c2 * nz(ss[2])
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 10, source_col: str = 'close') -> pd.Series:
        src = pd.to_numeric(df[source_col]).values
        n = len(src)
        
        # Coefficients
        a1 = math.exp(-1.414 * 3.14159 / period)
        # b1 unused in final formula in Pine script provided
        arg = 1.414 * 3.14159 / period
        c1 = 2 * a1 * math.cos(arg)
        c2 = -a1 * a1
        c3 = 1 - c1 - c2
        
        ss = np.zeros_like(src)
        
        # Recursive calculation
        # ss[i] = c3 * (src[i] + src[i-1])/2 + c1 * ss[i-1] + c2 * ss[i-2]
        
        for i in range(n):
            prev_src = src[i-1] if i > 0 else 0.0
            prev_ss1 = ss[i-1] if i > 0 else 0.0
            prev_ss2 = ss[i-2] if i > 1 else 0.0
            
            # Handling NaNs in source (treat as 0 or handled by loop?)
            # Usually strict numbers needed.
            if np.isnan(src[i]):
                 val_src = 0.0 # Or skip
            else:
                 val_src = src[i]
            
            if i == 0:
                 # Initial condition approximation
                 ss[i] = val_src 
            else:
                 ss[i] = c3 * (val_src + prev_src) / 2 + c1 * prev_ss1 + c2 * prev_ss2
                 
        return pd.Series(ss, index=df.index)
