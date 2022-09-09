from analyze_results import plot_auc_graphs




def main():
    plot_auc_graphs.graph_mapping['CirSqr'] = 'CircleSquareSrc'
    plot_auc_graphs.graph_mapping['CirSqrExt'] = 'CircleSquareDst'
    plot_auc_graphs.classes_to_evaluate['CirSqr'] = [0,1]
    plot_auc_graphs.classes_to_evaluate['CirSqrExt'] = [0,1]
    plot_auc_graphs.plot_graph(src='CirSqr',dst='CirSqrExt',seed=20,n_bootstraps=1000,graph_dir='example_code/auc_graphs/')
