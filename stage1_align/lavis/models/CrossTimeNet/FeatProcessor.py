# (seq_len, feat_dim, num_classes)
data_attribute = {
    'CR': (1197, 6, 12),
    'CT': (182, 3, 20),
    'EP': (206, 3, 4),
    'ER': (65, 4, 6),
    'FD': (62, 144, 2),
    'HB': (405, 61, 2),
    'JV': (29, 12, 9),
    'LI': (45, 2, 15),
    'NATOPS': (51, 24, 6),
    'PD': (8, 2, 10),
    'PEMS_SF': (144, 963, 7),
    'RS': (30, 6, 4),
    'SAD': (93, 13, 10),
    'SRS1': (896, 6, 2),
    'SRS2': (1152, 7, 2),
    'UWG': (315, 3, 8),
}

class FeatProcessor():
    def __init__(self):
        super(FeatProcessor, self).__init__()
    
    # def get_wave_length(self, input_list):
    #     output_list = [wave_dict[k] for k in input_list]
    #     return output_list
    
    # def get_seq_len(self, input_list):
    #     output_list = [seq_len_dict[k] for k in input_list]
    #     return output_list
    
    # def get_feat_dim(self, input_list):
    #     output_list = [feat_dim_dict[k] for k in input_list]
    #     return output_list
    
    # def get_num_classes(self, input_list):
    #     output_list = [num_classes_dict[k] for k in input_list]
    #     return output_list
    
    # def get_n_embd_num(self, input_list):
    #     output_list = [n_embd_num_dict[k] for k in input_list]
    #     return output_list
    
    def get_data_attribute(self, dataset_name, wave_rate):
        seq_len, feat_dim, num_classes = data_attribute[dataset_name]
        wave_length =  max(1, int(wave_rate * seq_len))
        
        return seq_len, feat_dim, num_classes, wave_length