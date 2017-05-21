from stats import *
import glob as gl
import pandas as pd
import os
import time

def printProgressBar (iteration, total, prefix, suffix = "completo",
         decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print("\n")
 

def run ( ):
     caminho_dados = "ArtificialDataset/*"
     header = 0
     sep = '\t'
     index_col = False
     
     
     dados = sorted(gl.glob(caminho_dados))
     dataset = caminho_dados.split('/')[-2]
     
     nomes = [ ]
     for i, base in enumerate(dados):
        nomes.append( base.split( '/' )[ -1 ].split( '.' )[ 0 ] )
     

     estatisticas = {'f1': f1_maximum_fisher_discriminating_ratio,
                     'f2': f2_volume_of_overlapping,
                     'f3': f3_maximum_individual_feat_efficiency,
                     'f4': f4_collective_reature_efficiency,
                     'l1': l1_sum_of_error_distance,
                     'l2': l2_rate_of_linear_classifier,
                     'n1': n1_fraction_borderline,
                     'n2': n2_ratio_intra_extra_class_nearest_neighbor_distance,
                     'n3': n3_error_rate_nearest_neighbor_classifier,
                     'l3': l3_non_linearity_of_linear_classifier,
                     'n4': n4_non_linearity_of_nearest_neighbor_classifier,
                     't1': t1_fraction_hyperspheres_covering_data,
                     't2': t2_average_number_of_examples_per_dimension}

     lista_stats = ['f1', 'f2', 'f3', 'f4', 'n1', 'n2', 'n3', 'n4', 'l1', 'l2', 'l3', 't1', 't2']
     results = pd.DataFrame()
     results[ 'stats' ] = lista_stats
     n = len(lista_stats) * len(nomes)
     j = 0
     
     # Output file, where the matched loglines will be copied to
     log_filename = os.path.normpath("logs/"+ dataset + ".log")
     output_filename = os.path.normpath("outputs/run_"+ dataset + ".csv")
     
     # Overwrites the file, ensure we're starting out with a blank file
     with open(log_filename, "w") as arquivo_log:  
         arquivo_log.write("stat,base,time,done\n")
         
     with open(output_filename, "w") as arquivo_out:
         arquivo_out.write("bases")
         for stat in lista_stats:
            arquivo_out.write( ",{0}".format(stat))
         arquivo_out.write( "\n" )
    
     start_all = time.time()
     for i, base in enumerate(dados):
        data = pd.read_csv( base, sep=sep,
            index_col=index_col, header=header)

        with open(output_filename, "a") as arquivo_out:
            arquivo_out.write( "{0}".format(nomes[i]) )
            
        stats = [ ]
        for k, stat in enumerate(lista_stats):
        
            with open(log_filename, "a") as arquivo_log:
                arquivo_log.write("{0},{1},".format(stat, nomes[i]))
            
            start = time.time()
            stats.append(estatisticas[stat](data))
            end = time.time()
            
            with open(output_filename, "a") as arquivo_out:
                arquivo_out.write(",{0}".format(stats[k]))
                
            with open(log_filename, "a") as arquivo_log:
                arquivo_log.write("{0},OK\n".format(end - start))
                
            j = j + 1
            printProgressBar(j, n-1, prefix = "{0} de {1}:".format(j, n-1))
            
        with open(output_filename, "a") as arquivo_out:
            arquivo_out.write("\n")
            
        results[ base ] = stats
     end_all = time.time()
     with open(log_filename, "a") as arquivo_log:
        arquivo_log.write("all,all,{0},OK\n".format(end_all - start_all))

     return results
     
if __name__ == "__main__":
    run()
