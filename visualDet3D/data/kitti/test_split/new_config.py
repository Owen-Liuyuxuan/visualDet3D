import numpy as np
train_lines = []
val_lines   = []

train_file  = 'train.txt'
val_file    = 'val.txt'
total_data_num = 7481

for i in range(total_data_num):
    i_string = "%06d\n" % i
    if np.random.rand() < 0.99:
        train_lines.append(i_string)
    else:
        val_lines.append(i_string)

with open(train_file, 'w') as file:
    file.writelines(train_lines)

with open(val_file, 'w') as file:
    file.writelines(val_lines)
