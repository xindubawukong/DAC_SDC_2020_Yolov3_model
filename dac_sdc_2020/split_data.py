# Split data to training set and validation set.


import os
import random

def main():

    dataset_path = '../../data_training'
    new_path = '../../dac_sdc_2020_dataset'
    train_ratio = 0.9  # valid ratio = 1 - train_ratio

    assert os.path.exists(dataset_path)
    if os.path.exists(new_path):
        os.system(f'rm -rf {new_path}')
    os.mkdir(new_path)
    train_path = os.path.join(new_path, 'train')
    valid_path = os.path.join(new_path, 'valid')
    os.mkdir(train_path)
    os.mkdir(valid_path)

    print(f'Splitting DAC SDC 2020 dataset from {dataset_path} to {new_path}.\n')

    all_data = []
    classes = []
    for cur_dir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(cur_dir, file)
                xml_path = os.path.join(cur_dir, file[:-4] + '.xml')
                label = cur_dir.split('/')[-1]
                all_data.append((image_path, xml_path, label))
                if label not in classes:
                    classes.append(label)
    print(f'The dataset contains {len(all_data)} images.\n')
    classes.sort()
    class_cnt = len(classes)
    print(f'There\'re {class_cnt} classes: {classes}.\n')

    # Generate "dac.names" file.
    class_names_path = os.path.join(new_path, 'dac.names')
    with open(class_names_path, 'w') as f:
        s = ''
        for class_name in classes:
            s += class_name + '\n'
        f.write(s)
        print(f'Class names written to {class_names_path}.\n')
    
    # Split randomly and copy files.
    random.shuffle(all_data)
    num = int(len(all_data) * train_ratio)
    train_data = all_data[:num]
    valid_data = all_data[num:]
    print('Copying data...')
    for id, (image_path, xml_path, label) in enumerate(train_data):
        image_path = '\'' + image_path + '\''
        xml_path = '\'' + xml_path + '\''
        new_image_path = os.path.join(train_path, 'train_' + str(id) + '_' + label + '.jpg')
        new_xml_path = os.path.join(train_path, 'train_' + str(id) + '_' + label + '.xml')
        os.system(f'cp {image_path} {new_image_path}')
        os.system(f'cp {xml_path} {new_xml_path}')
        print(f'Training data {id + 1}/{len(train_data)} completed.\r', end="")
    print('')
    for id, (image_path, xml_path, label) in enumerate(valid_data):
        image_path = '\'' + image_path + '\''
        xml_path = '\'' + xml_path + '\''
        new_image_path = os.path.join(valid_path, 'valid_' + str(id) + '_' + label + '.jpg')
        new_xml_path = os.path.join(valid_path, 'valid_' + str(id) + '_' + label + '.xml')
        os.system(f'cp {image_path} {new_image_path}')
        os.system(f'cp {xml_path} {new_xml_path}')
        print(f'Validation data {id + 1}/{len(valid_data)} completed.\r', end="")
    print('\nCompleted!')
    print(f'{len(train_data)} training images and xmls stored in {train_path}.')
    print(f'{len(valid_data)} validation images and xmls stored in {valid_path}.\n')

    # Final check.
    cnt = 0
    for cur_dir, dirs, files in os.walk(new_path):
        for file in files:
            if file.endswith('.jpg'):
                xml = os.path.join(cur_dir, file[:-4] + '.xml')
                assert os.path.exists(xml), xml
                cnt += 1
    assert cnt == len(all_data)
    print(f'{cnt} images and xml files copied successfully.\n')


if __name__ == '__main__':
    main()