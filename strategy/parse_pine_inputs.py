import re

with open('/home/vilivom/src/my/THC/pinescripts/regimline_v7.pine', 'r') as f:
    lines = f.readlines()

out = []
for line in lines:
    line = line.strip()
    if line.startswith('//') or not ('input.' in line or 'input(' in line):
        continue
    
    parts = line.split('=', 1)
    if len(parts) != 2:
        continue
    var_name = parts[0].strip()
    rest = parts[1].strip()
    
    # Try to extract default value
    # input.int(21, ...) -> 21
    # input.float(0.10, ...) -> 0.10
    # input.bool(true, ...) -> True
    
    m = re.search(r'input\.(int|float|bool|string)\(([^,]+)', rest)
    if m:
        default_val = m.group(2).strip()
        if default_val == 'true': default_val = 'True'
        elif default_val == 'false': default_val = 'False'
        
        # snake_case conversion for var_name
        # e.g. bearNeedLocalDowntrend -> bear_need_local_downtrend
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', var_name)
        snake_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
        print(f"self.{snake_name} = self.c.get('{var_name}', {default_val})")
    else:
        # custom like input.timeframe, or no type
        m2 = re.search(r'input\w*\(([^,]+)', rest)
        if m2:
            default_val = m2.group(1).strip()
            if default_val == 'true': default_val = 'True'
            elif default_val == 'false': default_val = 'False'
            # snake_case conversion
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', var_name)
            snake_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            print(f"self.{snake_name} = self.c.get('{var_name}', {default_val})")
        else:
            print(f"# COULD NOT PARSE: {line}")
