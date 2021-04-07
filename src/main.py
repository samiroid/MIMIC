import argparse
from randomseeds import run_analyses, run_tasks, extract_features

def get_args():
    par = argparse.ArgumentParser(description="MIMIC runs")
    par.add_argument('-input_path', type=str, required=True, help='input path')
    par.add_argument('-output_path',type=str, required=True, help='output path')    
    par.add_argument('-feats_path', type=str, required=True, help='features path')
    # par.add_argument('-cache_path', type=str, help='tmp path')
    par.add_argument('-dataset', type=str, required=True, help='dataset')
    par.add_argument('-feature_type', type=str, required=True, help='feature type')
    par.add_argument('-metric', type=str, required=True, help='feature type')
    par.add_argument('-clear_results', action="store_true", help='clear results cache')    
    par.add_argument('-tune',  action="store_true", help='tuning metric')
    par.add_argument('-mini_tasks', action="store_true", help='use small datasets')
    par.add_argument('-reset_tasks', action="store_true", help='re-run all tasks')    
    par.add_argument('-feature_extraction', action="store_true", help='extract features')
    par.add_argument('-subsample', action="store_true", help='extract features')
    
    return par.parse_args()  	

if __name__ == "__main__":
    args = get_args()    
    if args.feature_extraction:
        print("[extracting features: {} @ {}]".format(args.feature_type, args.feats_path))
        extract_features(feature_type=args.feature_type, path=args.feats_path)
    else:
        if "tasks" in args.dataset:
            print("[running tasks: {}]".format(args.dataset))
            print("input:{}\nfeature_type:{}\noutput:{}\nclear_results:{}\n".format(args.input_path, args.feature_type, args.output_path, args.clear_results))
            print("mini_tasks:{}\nreset_tasks:{}".format(args.mini_tasks, args.reset_tasks))                 

            run_tasks(args.input_path, args.dataset+".txt", args.feats_path, args.feature_type, args.output_path, args.metric, args.reset_tasks, args.mini_tasks)
        else:        
            print("input:{}\ndataset:{}\nfeature_type:{}\noutput:{}\nclear_results:{}\n".format(args.input_path, args.dataset, args.feature_type, args.output_path, args.clear_results))
        #     run_analyses(data_path, dataset, features_path, feature_type, results_path, 
        #         cache_path, args.metric,  clear_results=False)

            run_analyses(args.input_path, args.dataset, args.feats_path, args.feature_type, args.output_path, args.metric,args.clear_results, subsample=args.subsample)
