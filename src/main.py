import argparse
from MIMIC_Outcomes import gender_analysis, ethnicity_binary_analysis, ethnicity_analysis, clear_cache

def run_analyses(input_path, dataset, model, output_path, cache_path, clear_features, clear_results):
    if clear_features:
        clear_cache(cache_path, model=model, dataset=dataset,ctype="feats")
    if clear_results:
        clear_cache(cache_path, model=model, dataset=dataset,ctype="res*")
    
    gender_analysis(input_path, dataset, model, output_path, cache_path, plots=False)
    ethnicity_binary_analysis(input_path, dataset, model, output_path, cache_path, plots=False)
    ethnicity_analysis(input_path, dataset, model, output_path, cache_path, plots=False)


def get_args():
    par = argparse.ArgumentParser(description="MIMIC runs")
    par.add_argument('-input_path', type=str, required=True, help='input path')
    par.add_argument('-output_path',type=str, required=True, help='output path')
    par.add_argument('-cache_path', type=str, required=True, help='tmp path')
    par.add_argument('-dataset', type=str, required=True, help='dataset')
    par.add_argument('-model', type=str, required=True, help='tmp path')
    par.add_argument('-clear_results', action="store_true", help='clear results cache')
    par.add_argument('-clear_features', action="store_true", help='clear features cache')

    return par.parse_args()  	


if __name__ == "__main__":
    args = get_args()
    print("input:{}\ndataset:{}\nmodel:{}\noutput:{}\ncache:{}\nclear_feats:{}\nclear_results:{}\n".format(
        args.input_path, args.dataset, args.model, args.output_path, args.cache_path, 
        args.clear_features, args.clear_results))
    run_analyses(args.input_path, args.dataset, args.model, args.output_path, 
                args.cache_path, args.clear_features, args.clear_results)
