import os
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
folders = ['2', '3', '4']


def read_file():
    all_files = {}

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            all_files[folder] = files

    print(all_files)
    result = {}
    for folder, files in all_files.items():
        result[folder] = {}
        for filename in ['12.txt', '22.txt', '36.txt', '40.txt', '56.txt']:
            file_path = os.path.join(base_dir, folder, filename)
            values = []
            with open(file_path, 'r') as f:
                lines = f.readlines()[1:21]  # 2到21行，索引1到20
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        values.append(parts[5])
            result[folder][filename] = values
    print(result)
    return result

def draw_curve(dict_data):
    txt_files = ['12.txt', '22.txt', '36.txt', '40.txt', '56.txt']
    colors = {'2': 'r', '3': 'g', '4': 'b'}
    for txt in txt_files:
        plt.figure()
        for folder in dict_data:
            y = [int(v) for v in dict_data[folder][txt]]
            x = list(range(len(y)))
            plt.plot(x, y, label=f'Folder {folder}', color=colors.get(folder, None))
        plt.xlabel('Index')
        plt.ylabel('RSSI')
        plt.title(f'RSSI Curve for {txt}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{txt}_curve.png')
        
        plt.close()
if __name__ == '__main__':
    ret=read_file()
    draw_curve(ret)
    print("Done")