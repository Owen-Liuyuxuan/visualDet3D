
def process_train_val_file(split_file_path):

    with open(split_file_path) as f:
        train_lines = f.readlines()
        for i  in range(len(train_lines)):
            train_lines[i] = train_lines[i].strip()

    return train_lines