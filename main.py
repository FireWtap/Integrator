#!/usr/bin/env python3
"""
Single-Cell RNA Integration Benchmark Pipeline

A clean, modular pipeline for benchmarking scRNA-seq integration methods.
"""
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from preprocessing import (
    load_data, check_normalization_status, quality_control,
    preprocess_data, plot_batch_distribution
)
from integration import run_all_integrations
from benchmark import (
    benchmark_all_methods, plot_benchmark_results,
    plot_integration_umaps, save_results
)
from annotation import annotate_celltypist, list_celltypist_models


def main():
    parser = argparse.ArgumentParser(
        description='Single-cell RNA integration benchmark pipeline'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input h5ad file')
    parser.add_argument('--batch_key', type=str, required=True,
                       help='Column name for batch information')
    parser.add_argument('--label_key', type=str, default=None,
                       help='Column name for cell type labels')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['scgen', 'scanorama', 'harmony', 'combat', 'scvi'],
                       help='Integration methods to run')
    parser.add_argument('--n_hvg', type=int, default=2000,
                       help='Number of highly variable genes')
    parser.add_argument('--skip_qc', action='store_true',
                       help='Skip quality control plots')
    parser.add_argument('--annotate', action='store_true',
                       help='Run CellTypist annotation')
    parser.add_argument('--celltypist_model', type=str, default='Immune_All_Low.pkl',
                       help='CellTypist model to use')
    parser.add_argument('--list_models', action='store_true',
                       help='List available CellTypist models and exit')
    
    args = parser.parse_args()
    
    # List models and exit if requested
    if args.list_models:
        list_celltypist_models()
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("SINGLE-CELL RNA INTEGRATION BENCHMARK PIPELINE")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Batch key: {args.batch_key}")
    print(f"Label key: {args.label_key}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Annotation: {'Yes' if args.annotate else 'No'}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # 1. Load data
    adata = load_data(args.input)
    
    # 2. Check normalization status
    norm_status = check_normalization_status(adata)
    
    # 3. Quality control (before normalization)
    if not args.skip_qc:
        adata = quality_control(adata, output_dir=str(output_dir / "qc"))
        plot_batch_distribution(adata, args.batch_key, args.label_key,
                              output_dir=str(output_dir / "qc"))
    
    # 4. Normalize for CellTypist if needed (CellTypist requires log1p normalized to 10k)
    if args.annotate:
        print("\n=== Preparing data for CellTypist ===")
        adata_for_annotation = adata.copy()
        
        # CellTypist ALWAYS needs log1p normalized to 10k counts
        # Check if data is already in correct format
        if norm_status.get('likely_log_normalized', False):
            print("Data appears to be log-normalized already, using as-is for CellTypist...")
        elif norm_status.get('needs_normalization', True):
            print("Normalizing for CellTypist (10k counts + log1p)...")
            # Store raw counts before normalization
            adata_for_annotation.layers['counts'] = adata_for_annotation.X.copy()
            import scanpy as sc
            sc.pp.normalize_total(adata_for_annotation, target_sum=1e4)
            sc.pp.log1p(adata_for_annotation)
        else:
            # Data is in unknown state - normalize to be safe
            print("Data normalization unclear, normalizing for CellTypist (10k counts + log1p)...")
            adata_for_annotation.layers['counts'] = adata_for_annotation.X.copy()
            import scanpy as sc
            sc.pp.normalize_total(adata_for_annotation, target_sum=1e4)
            sc.pp.log1p(adata_for_annotation)
        
        # Run CellTypist annotation (batch-wise processing)
        adata_for_annotation = annotate_celltypist(
            adata_for_annotation, 
            model=args.celltypist_model,
            batch_key=args.batch_key,
            use_gpu=True
        )
        
        # Transfer annotations back to original adata
        if 'celltypist_predicted' in adata_for_annotation.obs:
            adata.obs['celltypist_predicted'] = adata_for_annotation.obs['celltypist_predicted']
            if 'celltypist_majority_voting' in adata_for_annotation.obs:
                adata.obs['celltypist_majority_voting'] = adata_for_annotation.obs['celltypist_majority_voting']
                adata.obs['celltypist_conf_score'] = adata_for_annotation.obs['celltypist_conf_score']
            
            # Use CellTypist predictions as label_key if not provided
            if args.label_key is None:
                args.label_key = 'celltypist_predicted'
                print(f"\n✓ Using CellTypist predictions as label_key: '{args.label_key}'")
                print(f"  This will be used for benchmarking integration quality")
            else:
                print(f"\n✓ CellTypist annotations added to adata.obs['celltypist_predicted']")
                print(f"  Using provided label_key: '{args.label_key}' for benchmarking")
        else:
            print("\n⚠ CellTypist annotation failed - no predictions available")
            if args.label_key is None:
                print("⚠ WARNING: No label_key provided and annotation failed")
                print("  Benchmarking will be skipped!")
        
        del adata_for_annotation
    
    # 5. Preprocessing for integration
    needs_norm = norm_status.get('needs_normalization', True)
    adata = preprocess_data(adata, normalize=needs_norm, hvg=True,
                           n_top_genes=args.n_hvg, scale=True)
    
    # 6. Run integration methods
    adata, integration_results, trained_models = run_all_integrations(
        adata, args.batch_key, args.label_key, methods=args.methods
    )
    
    # 7. Save trained models
    if trained_models:
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        print(f"\n=== Saving trained models ===")
        for method_name, model in trained_models.items():
            if model is not None:
                try:
                    model_path = models_dir / f"{method_name}_model"
                    model.save(str(model_path), overwrite=True)
                    print(f"✓ Saved {method_name} model to {model_path}")
                except Exception as e:
                    print(f"⚠ Failed to save {method_name} model: {e}")
    
    # 8. Benchmark
    if args.label_key and integration_results:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING WITH:")
        print(f"  Batch key: '{args.batch_key}'")
        print(f"  Label key: '{args.label_key}'")
        if args.label_key == 'celltypist_predicted':
            print(f"  (Using CellTypist auto-annotations)")
        print(f"{'='*60}")
        
        results_df, integrated_adatas = benchmark_all_methods(
            adata, args.batch_key, args.label_key, integration_results
        )
        
        # 9. Visualize results
        plot_benchmark_results(results_df, output_dir=str(output_dir))
        plot_integration_umaps(adata, integrated_adatas, args.batch_key,
                             args.label_key, output_dir=str(output_dir))
        
        # 10. Save results
        save_results(results_df, adata, output_dir=str(output_dir))
    else:
        if not args.label_key:
            print("\n⚠ No label_key provided - skipping benchmarking")
        if not integration_results:
            print("\n⚠ No integration methods succeeded - skipping benchmarking")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
