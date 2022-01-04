from models import *
from dataset import *
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm


#prediction

if not os.path.exists('output_csv'):
    os.mkdir('output_csv')

parser = argparse.ArgumentParser(description='Train SimTriplet')
parser.add_argument('--data', default='./test_npy/', type=str, help='test data directory')
parser.add_argument('--output_filename', default='./result')
parser.add_argument('--csv_filepath', default='./threshold.csv', type=str, help='default: threshold.csv')

args = parser.parse_args()
test_dir = args.data
output_filename = args.output_filename
csv_filepath = args.csv_filepath

csv = pd.read_csv(csv_filepath)
best_thrd = csv['threshold'][0]    ###CHECKHERE
print(best_thrd)

model_path = './model.pth'

model = SimTriplet()
save_dict = torch.load(model_path)
model.load_state_dict(save_dict)
model = model.cuda()

query_path = f'./queries.csv'
output_csv = f'./{output_filename}.csv'


test_dataset = TestDataset(query_path, test_dir)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle = False
)

# best_thrd = 0.21452296973796803

model.eval()
output = pd.DataFrame(columns= ['query', 'prediction'])
with torch.no_grad():
    for img1, img2, query in tqdm(test_dataloader):
        img1, img2 = img1.cuda(), img2.cuda()
        feature1, p1= model(img1)
        feature2, p2 = model(img2)
        cos_sim = F.cosine_similarity(feature1, feature2)
        cos_sim = cos_sim.detach().cpu().numpy()
        if(cos_sim > best_thrd):
            temp = pd.DataFrame({
                'query': query,
                'prediction': 1
            }, index=[0])
        else:
            temp = pd.DataFrame({
                'query': query,
                'prediction': 0
            }, index=[0])
        output = output.append(temp, ignore_index=True)

output.to_csv(output_csv, index=False)
