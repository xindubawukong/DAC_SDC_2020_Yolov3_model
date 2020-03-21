import os


dataset_path = '../../data_training'
goto_path = '../../dac_sdc_2020_dataset/all'

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
all_data.sort()
classes.sort()
class_cnt = len(classes)
print(f'There\'re {class_cnt} classes: {classes}.\n')

cnt = 0
for ii, (jpg, xml, label) in enumerate(all_data):
    jpg = '\'' + jpg + '\''
    xml = '\'' + xml + '\''
    new_jpg = os.path.join(goto_path, str(cnt) + '_' + label + '.jpg')
    os.system(f'cp {jpg} {new_jpg}')
    new_xml = os.path.join(goto_path, str(cnt) + '_' + label + '.xml')
    os.system(f'cp {xml} {new_xml}')
    cnt += 1
    print(f'All data {ii + 1}/{len(all_data)} completed.\r', end="")
print('\nCompleted!')