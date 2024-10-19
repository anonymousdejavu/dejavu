# todo: load dummy dataset, run throughput
import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, num_samples, same_data=False):
        self.num_samples = num_samples
        self.onehot_reference_type = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 2])) # 4,3
        self.onehot_reference_type = self.onehot_reference_type.view(4, 3, 1, 1, 1).expand(4, 3, 4, 14, 14)
        
        self.same_data = same_data
        if self.same_data:
            self.pixel_data = torch.randn(4,3,224,224)
            self.compressed_data = torch.randn(4,3,4,14,14)
            self.compressed_data = torch.cat((self.compressed_data, self.onehot_reference_type), dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Generate random data for each sample
        if self.same_data:
            return self.pixel_data, self.compressed_data
        pixel_data = torch.randn(4,3,224,224)
        compressed_data = torch.randn(4,3,4,14,14)
        compressed_data = torch.cat((compressed_data, self.onehot_reference_type), dim=1)
        return pixel_data, compressed_data

    def get_first_cache(self, batch_size):
        cached_data = torch.zeros(12,5,batch_size,3,197,768)
        # cached_data[:,4] /= cached_data[:,4].norm(dim=-1, keepdim=True)
        return cached_data


def post_proc_dummy(dl_output):
    batch_pixel_data, batch_compressed_data = dl_output
    pixel_shape = batch_pixel_data.shape[2:] # C,H,W
    batch_pixel_data = batch_pixel_data.transpose(0,1).reshape(-1, *pixel_shape)
    batch_compressed_data = batch_compressed_data.transpose(0,1)

    return batch_pixel_data, batch_compressed_data


if __name__=="__main__":
    # Example usage:
    batch_size = 16
    num_samples = 100

    dummy_dataset = DummyDataset(num_samples=num_samples)

    # Create a data loader
    data_loader = DataLoader(dataset=dummy_dataset, batch_size=batch_size, shuffle=True)

    # Iterate through the data loader
    for batch in data_loader:
        batch_pixel_data, batch_compressed_data = post_proc_dummy(batch)
        print("Batch shape:", batch_pixel_data.shape, batch_compressed_data.shape)
        break  # Break after the first batch for demonstration
