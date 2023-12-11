import os
import csv
import argparse

from torch.utils.data import Dataset, DataLoader

class dataloader(Dataset):
    def __init__(self, samples_path, csv_path, hiyari_folder_path, config_path):
        self.generate_testcase_file(A=samples_path, B=csv_path, main_path=hiyari_folder_path, config_txt=config_path)
        self.config_path = config_path
        self.data = self.read_config_file()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, video_path, csv_path = self.data[idx]
        return file_name, video_path, csv_path
    
    def generate_testcase_file(self, A, B, main_path, config_txt):
        if not os.path.exists(main_path):
            print(f"Error: Hiyari folder '{main_path}' does not exist.")
            return

        video_folder = os.path.join(main_path, A)
        csv_folder = os.path.join(main_path, B)

        if not os.path.exists(video_folder):
            print(f"Error: Folder '{video_folder}' does not exist.")
            return

        if not os.path.exists(csv_folder):
            print(f"Error: Folder '{csv_folder}' does not exist.")
            return

        files_in_A = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]

        with open(os.path.join(self.results_folder, config_txt), 'w') as testcase_file:
            testcase_file.write("video_path, csv_path\n")
            for file_name in files_in_A:
                video_path = os.path.join(video_folder, file_name)
                csv_path = os.path.join(csv_folder, os.path.splitext(file_name)[0] + '.csv')
                testcase_file.write(f"{file_name}, {video_path}, {csv_path}\n")
    
    def read_config_file(self):
        data = []
        with open(self.config_path, 'r') as file:
            next(file)
            for line in file:
                file_name, video_path, csv_path = line.strip().split(', ')
                data.append({
                    'file_name': file_name,
                    'video_path': video_path,
                    'csv_path': csv_path
                })
        return data

# custom_dataset = CustomDataset(txt_file=txt_file_path)
# dataloader = DataLoader(dataset=custom_dataset, batch_size=1, shuffle=True)

# for batch in dataloader:
#     file_name, video_path, csv_path = batch
#     print(f"File Name: {file_name}, Video Path: {video_path}, CSV Path: {csv_path}")

class VehicleExtraction:
    def __init__(self):
        return
    def CSV_reader_Hiyari(self, loadpath, savepath):
        velocity = []
        
        with open(loadpath, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            for row in csv_reader:
                vec = float(row[3])*3.6
                velocity.append(vec)

        with open(savepath, 'w') as txt_file:
            for val in velocity:
                txt_file.write(str(val)+'\n')    
        
        print('Extract velocity to file to file: ',savepath)
        return

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', help='')
parser.add_argument('--save_path', help='')
args = parser.parse_args()

if __name__ == '__main__':
    loader = VehicleExtraction()
    loader.CSV_reader_Hiyari(args.load_path, args.save_path)

