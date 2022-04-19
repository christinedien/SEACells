from tqdm import tqdm
import pyranges as pr
from sklearn.metrics import pairwise_distances
from scipy.stats import rankdata

import numpy as np
import pandas as pd
import scanpy as sc

from . import core


def prepare_multiome_anndata(atac_ad, rna_ad, SEACells_label='SEACell', n_bins_for_gc=50):
    """
    Function to create metacell Anndata objects from single-cell Anndata objects for multiome data

    :param atac_ad: (Anndata) ATAC Anndata object with raw peak counts in `X`. These anndata objects should be constructed 
     using the example notebook available in 
    :param rna_ad: (Anndata) RNA Anndata object with raw gene expression counts in `X`. Note: RNA and ATAC anndata objects 
     should contain the same set of cells
    :param SEACells_label: (str) `atac_ad.obs` field for constructing metacell matrices. Same field will be used for 
      summarizing RNA and ATAC metacells. 
    :param n_bins_gc: (int) Number of bins for creating GC bins of ATAC peaks.

    :return: ATAC metacell Anndata object and RNA metacell Anndata object
    """

    # Subset of cells common to ATAC and RNA
    common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names)
    if len(common_cells) != atac_ad.shape[0]:
        print('Warning: The number of cells in RNA and ATAC objects are different. Only the common cells will be used.')
    atac_mod_ad = atac_ad[common_cells, :]
    rna_mod_ad = rna_ad[common_cells, :]

    # #################################################################################
    # Generate metacell matrices

    # Set of metacells
    metacells = atac_mod_ad.obs[SEACells_label].astype(str).unique()
    metacells = metacells[atac_mod_ad.obs[SEACells_label].value_counts()[
        metacells] > 1]

    print('Generating Metacell matrices...')
    print(' ATAC')
    atac_meta_ad = core.summarize_by_SEACell(atac_mod_ad, SEACells_label=SEACells_label, summarize_layer='X')
    atac_meta_ad = atac_meta_ad[metacells, :]
    # ATAC - Summarize SVD representation 

    svd = pd.DataFrame(atac_mod_ad.obsm['X_svd'], index=atac_mod_ad.obs_names)
    summ_svd = svd.groupby(atac_mod_ad.obs[SEACells_label]).mean()
    atac_meta_ad.obsm['X_svd'] = summ_svd.loc[atac_meta_ad.obs_names, :].values


    # ATAC - Normalize
    _add_atac_meta_data(atac_meta_ad, atac_mod_ad, n_bins_for_gc)
    sc.pp.filter_genes(atac_meta_ad, min_cells=1)
    _normalize_ad(atac_meta_ad)

    # RNA summaries using ATAC SEACells
    print(' RNA')
    rna_mod_ad.obs['temp'] = atac_mod_ad.obs[SEACells_label]
    rna_meta_ad = core.summarize_by_SEACell(rna_mod_ad, SEACells_label='temp', summarize_layer='X')
    rna_meta_ad = rna_meta_ad[metacells, :]
    _normalize_ad(rna_meta_ad)

    return atac_meta_ad, rna_meta_ad


def prepare_integrated_anndata(atac_ad, rna_ad, mapping, SEACells_label='SEACell', n_bins_for_gc=50):
    """
    Function to create metacell Anndata objects from single-cell Anndata objects for integrated RNA and ATAC data

    :param atac_ad: (Anndata) ATAC Anndata object with raw peak counts in `X`. These anndata objects should be constructed 
     using the example notebook available in 
    :param rna_ad: (Anndata) RNA Anndata object with raw gene expression counts in `X`. Note: RNA and ATAC anndata objects 
     should contain the same set of cells
    :param mapping: TBD
    :param SEACells_label: (str) `atac_ad.obs` field for constructing metacell matrices. Same field will be used for 
      summarizing RNA and ATAC metacells. 
    :param n_bins_gc: (int) Number of bins for creating GC bins of ATAC peaks.

    :return: ATAC metacell Anndata object and RNA metacell Anndata object
    """

    # Copy to leave the original, raw AnnDatas unmodified:
    atac_mod_ad = atac_ad.copy()
    rna_mod_ad = rna_ad.copy()
    
    # #################################################################################
    # Generate metacell matrices
    # Since the Mapping was made off of the RNA metacells, there may be dupliated ATAC
    #    metacells. For this reason, the RNA metacells will be used to define the pairs
    #    of metacells and their common name
    
    # Metacell set
    rna_metacells = rna_mod_ad.obs[SEACells_label].astype(str).unique()
    rna_metacells = rna_metacells[rna_mod_ad.obs[SEACells_label].value_counts()[rna_metacells] > 1]

    # Mapping beetween RNA and ATAC
    mapping = mapping.loc[rna_metacells]
    # Create common metacell name
    mapping['common'] = np.arange(len(mapping))
    mapping['common'] = 'metacell ' + mapping['common'].astype(str)
    
    print('Generating Metacell matrices...')
    print(' RNA')
   
    # RNA - Summary matrix 
    summ_matrix = pd.DataFrame(0.0, index=mapping['common'], columns=rna_mod_ad.var_names)
    for m in tqdm(summ_matrix.index):
        rna_meta = mapping.index[mapping['common'] == m][0]
        
        cells = rna_mod_ad.obs_names[rna_mod_ad.obs[SEACells_label] == rna_meta]
        summ_matrix.loc[m, :] = np.ravel(rna_mod_ad[cells, :].X.sum(axis=0))

    # RNA - create metacell matrix
    rna_meta_ad = _create_ad(summ_matrix)
    rna_meta_ad.obs['original_rna'] = rna_metacells

    # RNA- Normalize 
    _normalize_ad(rna_meta_ad)
    
    print(' ATAC')
    # ATAC - Summarize by metacells
    # Summary matrix
    summ_matrix = pd.DataFrame(0.0, index=mapping['common'], columns=atac_mod_ad.var_names)
    svd = pd.DataFrame(atac_mod_ad.obsm['X_svd'], index=atac_mod_ad.obs_names)
    meta_svd = pd.DataFrame(0.0, index=mapping['common'], columns=svd.columns)
    for m in tqdm(summ_matrix.index):
        atac_metacell = mapping.loc[mapping['common']== m, 'atac'].item()
        cells = atac_mod_ad.obs_names[atac_mod_ad.obs[SEACells_label] == atac_metacell]
        summ_matrix.loc[m, :] = np.ravel(atac_mod_ad[cells, :].X.sum(axis=0))
        meta_svd.loc[m, :] = svd.loc[cells, :].sum(axis=0)

    # ATAC - create metacell anndata
    atac_meta_ad = _create_ad(summ_matrix)
    atac_meta_ad.obsm['X_svd'] = meta_svd
    atac_meta_ad.obs['original_atac'] = mapping['atac'].values
    
    # ATAC - Normalize
    _add_atac_meta_data(atac_meta_ad, atac_mod_ad, n_bins_for_gc) 
    sc.pp.filter_genes(atac_meta_ad, min_cells=1)
    _normalize_ad(atac_meta_ad)
    
    return atac_meta_ad, rna_meta_ad


def _create_ad(summ_matrix):
    from scipy.sparse import csr_matrix

    meta_ad = sc.AnnData(summ_matrix)
    meta_ad.X = csr_matrix(meta_ad.X)
    meta_ad.obs_names, meta_ad.var_names = summ_matrix.index.astype(str), summ_matrix.columns

    return meta_ad

def _normalize_ad(meta_ad, save_raw=True):
    if save_raw:
        # Save in raw
        meta_ad.raw = meta_ad.copy()

    # Normalize 
    sc.pp.normalize_total(meta_ad, key_added='n_counts')
    sc.pp.log1p(meta_ad)
    

def _add_atac_meta_data(atac_meta_ad, atac_ad, n_bins_for_gc):
    atac_ad.var['log_n_counts'] = np.ravel(np.log10(atac_ad.X.sum(axis=0)))
    
    atac_meta_ad.var['GC_bin'] = np.digitize(atac_ad.var['GC'], np.linspace(0, 1, n_bins_for_gc))
    atac_meta_ad.var['counts_bin'] = np.digitize(atac_ad.var['log_n_counts'],
                                                 np.linspace(atac_ad.var['log_n_counts'].min(),
                                                             atac_ad.var['log_n_counts'].max(), 
                                                             n_bins_for_gc))

def _pyranges_from_strings(pos_list):
    """
    Function to create pyranges for a `pd.Series` of strings
    """
    # Chromosome and positions
    chr = pos_list.str.split(':').str.get(0)
    start = pd.Series(pos_list.str.split(':').str.get(1)
                      ).str.split('-').str.get(0)
    end = pd.Series(pos_list.str.split(':').str.get(1)
                    ).str.split('-').str.get(1)

    # Create ranges
    gr = pr.PyRanges(chromosomes=chr, starts=start, ends=end)
    return gr


def _pyranges_to_strings(peaks):
    """
    Function to convert pyranges to `pd.Series` of strings of format 'chr:start-end'
    """
    # Chromosome and positions
    chr = peaks.Chromosome.astype(str).values
    start = peaks.Start.astype(str).values
    end = peaks.End.astype(str).values

    # Create ranges
    gr = chr + ':' + start + '-' + end

    return gr


def load_transcripts(path_to_gtf):
    """
    Load transcripts from GTF File. `chr` is preprended to each entry
    """
    gtf = pr.read_gtf(path_to_gtf)
    gtf.Chromosome = 'chr' + gtf.Chromosome.astype(str)
    transcripts = gtf[gtf.Feature == 'transcript']
    return transcripts


def _peaks_correlations_per_gene(gene,
                                 atac_exprs,
                                 rna_exprs,
                                 atac_meta_ad,
                                 peaks_pr,
                                 transcripts,
                                 span,
                                 n_rand_sample=100):

    # Gene transcript - use the longest transcript
    gene_transcripts = transcripts[transcripts.gene_name == gene]
    if len(gene_transcripts) == 0:
        return 0
    longest_transcript = gene_transcripts[
        np.arange(len(gene_transcripts)) == np.argmax(gene_transcripts.End - gene_transcripts.Start)]
    start = longest_transcript.Start.values[0] - span
    end = longest_transcript.End.values[0] + span

    # Gene span
    gene_pr = pr.from_dict({'Chromosome': [longest_transcript.Chromosome.values[0]],
                            'Start': [start],
                            'End': [end]})
    gene_peaks = peaks_pr.overlap(gene_pr)
    if len(gene_peaks) == 0:
        return 0
    gene_peaks_str = _pyranges_to_strings(gene_peaks)

    # Compute correlations
    X = atac_exprs.loc[:, gene_peaks_str].T
    cors = 1 - np.ravel(pairwise_distances(np.apply_along_axis(rankdata, 1, X.values),
                                           rankdata(rna_exprs[gene].T.values).reshape(
                                               1, -1),
                                           metric='correlation'))
    cors = pd.Series(cors, index=gene_peaks_str)

    # Random background
    df = pd.DataFrame(1.0, index=cors.index, columns=['cor', 'pval'])
    df['cor'] = cors
    
    from scipy.stats import norm
    for p in df.index:
        try:
            # Try random sampling without replacement
            rand_peaks = np.random.choice(atac_meta_ad.var_names[(atac_meta_ad.var['GC_bin'] == atac_meta_ad.var['GC_bin'][p]) &
                                                                 (atac_meta_ad.var['counts_bin'] == atac_meta_ad.var['counts_bin'][
                                                                     p])], n_rand_sample, False)
        except:
            rand_peaks = np.random.choice(atac_meta_ad.var_names[(atac_meta_ad.var['GC_bin'] == atac_meta_ad.var['GC_bin'][p]) &
                                                                 (atac_meta_ad.var['counts_bin'] == atac_meta_ad.var['counts_bin'][
                                                                     p])], n_rand_sample, True)

        if type(atac_exprs) is sc.AnnData:
            X = pd.DataFrame(atac_exprs[:, rand_peaks].X.todense().T)
        else:
            X = atac_exprs.loc[:, rand_peaks].T

        # For p-value calculation, remove the peaks with no variation across cells
        X = X.iloc[np.where(np.std(X.values, axis=1) !=0)]
        if X.shape[0] != 0:
 
            rand_cors = 1 - np.ravel(pairwise_distances(np.apply_along_axis(rankdata, 1, X.values),
                                                        rankdata(rna_exprs[gene].T.values).reshape(1, -1),
                                                        metric='correlation'))

            m = np.mean(rand_cors)
            v = np.std(rand_cors)

            # If the standard deviation is zero, set a p-value of 1
            if v != 0:
                df.loc[p, 'pval'] = (norm.cdf(-np.abs(cors[p]), m, v)) + (1 - norm.cdf(np.abs(cors[p]), m, v))
    return df


def get_gene_peak_correlations(atac_meta_ad,
                               rna_meta_ad,
                               path_to_gtf,
                               span=100000,
                               n_jobs=1,
                               gene_set=None):
    """
    Function to compute  correlations between gene expression and peak accessibility

    :param atac_meta_ad: (Anndata) ATAC metacell Anndata created using `prepare_multiome_anndata`
    :param rna_meta_ad: (Anndata) RNA metacell Anndata created using `prepare_multiome_anndata`
    :param path_to_gtf: (str) Path to ENSEMBL GTF file
    :param span: (int) Genomic window around the gene body to identify for which correlations with expression are computed
    :param n_jobs: (int) Number of jobs for parallel processing
    :param gene_set: (pd.Series) Subset of genes for which to compute correlations. All genes are used by default

    :return: `pd.Series` with a dataframe of correlation and p-value for each gene. Note that p-value is one-sided assuming positive correlations
    """

    # #################################################################################
    print('Loading transcripts per gene...')
    transcripts = load_transcripts(path_to_gtf)

    print('Preparing matrices for gene-peak associations')
    atac_exprs = pd.DataFrame(atac_meta_ad.X.todense(),
                              index=atac_meta_ad.obs_names, columns=atac_meta_ad.var_names)
    rna_exprs = pd.DataFrame(rna_meta_ad.X.todense(),
                             index=rna_meta_ad.obs_names, columns=rna_meta_ad.var_names)
    peaks_pr = _pyranges_from_strings(atac_meta_ad.var_names)

    print('Computing peak-gene correlations')
    if gene_set is None:
        use_genes = rna_meta_ad.var_names
    else:
        use_genes = gene_set
    from joblib import Parallel, delayed
    gene_peak_correlations = Parallel(n_jobs=n_jobs)(delayed(_peaks_correlations_per_gene)(gene,
                                                                                      atac_exprs,
                                                                                      rna_exprs,
                                                                                      atac_meta_ad,
                                                                                      peaks_pr,
                                                                                      transcripts,
                                                                                      span)
                                                for gene in tqdm(use_genes))
    gene_peak_correlations = pd.Series(gene_peak_correlations, index=use_genes)
    return gene_peak_correlations


def _get_sig_peaks(df, min_corr=-1.0, max_corr=1.0,
                   min_pval=0.0, max_pval=0.1, incl='both'):
    """
    TODO: Documentation
    """
    sig_peaks = df.index[(df['pval'].between(min_pval, max_pval,inclusive=incl)) & (
                          df['cor'].between(min_corr, max_corr, inclusive=incl))].tolist()

    return sig_peaks


def get_peak_counts(gene_peak_correlations, min_corr=-1.0, max_corr=1.0,
                    min_pval=0.0, max_pval=0.1, incl='both', return_peaks=False):
    """
    Determine the number of significantly correlated peaks per gene

    :param gene_peak_correlations: (pd.Series) Output of `get_gene_peak_correlations` function
   
    :return: `pd.Series` with number of significantly positive correlated peaks with each gene
    """
    peak_counts = pd.Series(0, index=gene_peak_correlations.index)
    for gene in tqdm(peak_counts.index):
        df = gene_peak_correlations[gene]
        if type(df) is int:
            continue

        sig_peaks = _get_sig_peaks(df, min_corr, max_corr, min_pval, max_pval, incl)
        
        if return_peaks and (len(sig_peaks) != 0):
            peak_counts[gene] = sig_peaks
        else:
            peak_counts[gene] = len(sig_peaks)
    
    return peak_counts


def get_gene_scores(atac_meta_ad, gene_peak_correlations, min_corr=-1.0, max_corr=1.0,
                    min_pval=0.0, max_pval=0.1, incl='both'):
    """
    Compute the aggregate accessibility of all peaks associated with each gene. G
    ene scores are computed as the aggregate accessibility of all the signficantly correlated peaks associated with a gene.

    :param atac_meta_ad: (Anndata) ATAC metacell Anndata created using `prepare_multiome_anndata`
    :param gene_peak_correlations: (pd.Series) Output of `get_gene_peak_correlations` function
    :param p_val_cutoff: (float) Nominal p-value cutoff for test of significance of correlation
    :param cor_cutoff: (float) Correlation cutoff

    :return: `pd.DataFrame` of ATAC gene scores (cells X genes)
    """
    gene_scores = pd.DataFrame(0.0, index=atac_meta_ad.obs_names, columns=gene_peak_correlations.index)

    for gene in tqdm(gene_scores.columns):
        df = gene_peak_correlations[gene]
        if type(df) is int:
            continue
        gene_peaks = _get_sig_peaks(df, min_corr, max_corr, min_pval, max_pval, incl)
        gene_scores[gene] = np.ravel(np.dot(atac_meta_ad[:, gene_peaks].X.todense(),
                                            df.loc[gene_peaks, 'cor']))
    gene_scores = gene_scores.loc[:, (gene_scores.sum() >= 0)]
    return gene_scores
