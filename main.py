import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from utils import *
from logger import Logger
import os.path as osp
from torch_geometric.datasets import Planetoid
from model import *
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def train(model, predictor, data, split_edge, optimizer, batch_size):
    
    predictor.train()
    model.train()
    
    
    pos_train_edge = split_edge['train']['edge'].to(data.edge_index.device)
    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(data.x, data.adj_t, data.edge_weight)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long, device=data.x.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()



def test(model, predictor, data, split_edge, evaluator, batch_size, id_to_name=None):
    predictor.eval()
    model.eval()

    h = model(data.x, data.adj_t, data.edge_weight)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    def get_preds(edge_tensor):
        preds = []
        for perm in DataLoader(range(edge_tensor.size(0)), batch_size):
            edge = edge_tensor[perm].t()
            preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        return torch.cat(preds, dim=0)

    pos_train_pred = get_preds(pos_train_edge)
    pos_valid_pred = get_preds(pos_valid_edge)
    neg_valid_pred = get_preds(neg_valid_edge)
    pos_test_pred = get_preds(pos_test_edge)
    neg_test_pred = get_preds(neg_test_edge)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        

    train_result = torch.cat((torch.ones_like(pos_train_pred), torch.zeros_like(neg_valid_pred)), dim=0)
    train_pred = torch.cat((pos_train_pred, neg_valid_pred), dim=0)

    valid_result = torch.cat((torch.ones_like(pos_valid_pred), torch.zeros_like(neg_valid_pred)), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (
        roc_auc_score(train_result.numpy(), train_pred.numpy()),
        roc_auc_score(valid_result.numpy(), valid_pred.numpy()),
        roc_auc_score(test_result.numpy(), test_pred.numpy()),
    )

    results['AUPRC'] = (
        average_precision_score(train_result.numpy(), train_pred.numpy()),
        average_precision_score(valid_result.numpy(), valid_pred.numpy()),
        average_precision_score(test_result.numpy(), test_pred.numpy()),
    )

    


    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--mlp_num_layers', type=int, default=11)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--init', type=str, choices=['SGC', 'RWR', 'KI', 'Random', 'WS', 'Null'], default='KI')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_type', type=str, default='Specific')
    parser.add_argument('--data_name', type=str, default='hESC')
    parser.add_argument('--data_num', type=int, default='500')
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()
    print(args)
    
    seed = args.seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    path = "/File Path" + args.data_type + " Dataset/" + args.data_name + "/TFs+" + str(args.data_num) + "/BL--ExpressionData.csv"
    df = pd.read_csv(path , index_col = 0)
    node_feature_matrix = df.values
    feature_embeddings = torch.from_numpy(node_feature_matrix).float()
    
    path = "/File Path" + args.data_type + " Dataset/" + args.data_name + "/TFs+" + str(args.data_num) + "/Label.csv"
    edge_df = pd.read_csv(path , index_col = 0) 
 
    source_nodes = edge_df['TF'].values
    target_nodes = edge_df['Target'].values

   
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    reverse_edge_index = edge_index[[1, 0]]  
    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)  

   
    edge_index = torch.unique(edge_index, dim=1)
    
    edge_attr = []
    for i in range(edge_index.shape[1]):
        source_node = edge_index[0, i]
        target_node = edge_index[1, i]
        source_attr = feature_embeddings[source_node]
        target_attr = feature_embeddings[target_node]
        average_attr = (source_attr + target_attr) / 2
        edge_attr.append(average_attr)

    edge_attr = torch.stack(edge_attr)

    
    data = Data(x=feature_embeddings, edge_index=edge_index, edge_attr=edge_attr)
    
   
    
    
    split_edge = {
    'train': {
        'edge': [],        
        'edge_neg': []     
    },
    'valid': {
        'edge': [],        
        'edge_neg': []     
    },
    'test': {
        'edge': [],        
        'edge_neg': []     
    }
                }
    
    for split in ['Test', 'Validation', 'Train']:
        path = "/File Path" + args.data_type + "/" + args.data_name + ' ' + str(args.data_num) + '/' + split + '_set.csv'
        df = pd.read_csv(path, index_col = 0)
        negative_edges = df[df['Label'] == 0][['TF' , 'Target']].values
        postive_edges = df[df['Label'] == 1][['TF' , 'Target']].values
        if split == 'Test':
            split_edge['test']['edge_neg'].extend([(int(u) , int(v)) for u, v in negative_edges])
            split_edge['test']['edge'].extend([(int(u) , int(v)) for u, v in postive_edges])
        if split == 'Validation':
            split_edge['valid']['edge_neg'].extend([(int(u) , int(v)) for u, v in negative_edges])
            split_edge['valid']['edge'].extend([(int(u) , int(v)) for u, v in postive_edges])
        if split == 'Train':
            split_edge['train']['edge_neg'].extend([(int(u) , int(v)) for u, v in negative_edges])
            split_edge['train']['edge'].extend([(int(u) , int(v)) for u, v in postive_edges])
            
    for key in ['train', 'valid', 'test']:
        for sub_key in ['edge', 'edge_neg']:
            edge_list = split_edge[key][sub_key]
            edge_tensor = torch.tensor(edge_list, dtype=torch.long)
            split_edge[key][sub_key] = edge_tensor
    
    
    

    data.edge_index = split_edge['train']['edge'].t()
    
   
    
    data = T.ToSparseTensor(remove_edge_index=False)(data)
    data = data.to(device)
    
 
    
    
 
    path = "/File Path" + args.data_type + " Dataset/" + args.data_name + "/TFs+" + str(args.data_num) + "/Target.csv"
    target_df = pd.read_csv(path)
    path = "/File Path" + args.data_type + " Dataset/" + args.data_name + "/TFs+" + str(args.data_num) + "/TF.csv"
    tf_df = pd.read_csv(path)


    combined_df = pd.concat([
        tf_df[['index', 'TF']].rename(columns={'TF': 'name'}),
        target_df[['index', 'Gene']].rename(columns={'Gene': 'name'})
    ])

  
    combined_df = combined_df.drop_duplicates(subset='index', keep='first')

    id_to_name = dict(zip(combined_df['index'], combined_df['name']))
   
    
   
    model = scHGNN(data, args).to(device)
    predictor = LinkPredictor(data.num_features, args.hidden_channels, 1, args.mlp_num_layers, args.dropout).to(device)
    para_list = list(model.parameters()) + list(predictor.parameters())
    total_params = sum(p.numel() for param in para_list for p in param)
    total_params_print = f'Total number of model parameters is {total_params}'
    
    evaluator = Evaluator(name='ogbl-collab')
    
    loggers = {
        'AUC': Logger(args.runs, args),
        'AUPRC': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(predictor.parameters()) + list(model.parameters()), lr=args.lr)
        
        start_time = time.time()
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator, args.batch_size,id_to_name)
                
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    spent_time = time.time() - start_time
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')
                    print(f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                    print('---')
                    start_time = time.time()

        
        

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
