import java.util.stream.Collectors;

// Workaround for allowing comma separated list of rank upper/lower
params.rankslist = params.ranks?.split(',') as List

process biCVSplit {
    // Shuffle data matrix and split into 9 submatrices for bi-cross validation
    label 'canBeLocal'
    input:
        path matrix
    output:
        path 'shuffle_*.npz'
    script:
        """
        cva_split.py -i $params.n -m $matrix -s $params.seed
        """
    stub:
        """
        for i in \$(seq 1 10);
        do
            touch shuffle_${i}.npz
        done
        """
}

process rankBiCv {
    // Perform bi-fold cross validation for a given rank and matrix
    // When multiple inputs, each input must come from a different channel
    // Here this is okay, as we will make a channel which emits the desired 
    // ranks to search.
    input:
        tuple val(rank), path(x)
    // The results file has all the params in it, so process which joins
    // can use that instead
    output:
        path('results.pickle')
    cpus 4
    script:
        """
        cva_rank.py \
        --folds $x \
        --seed $params.seed \
        --rank $rank \
        --max_iter $params.nmf_max_iter \
        --verbose
        """
    stub:
        """
        touch results.pickle
        """
}

process rankCombineScores {
    label 'canBeLocal'
    input:
        path(results, stageAs: "results*.pickle")
    // We will publish this as it is one of the primary outputs
    publishDir { params.publish_dir }, mode: 'copy', overwrite: true
    // The output file here will have the rank as a column, so we can just pass
    // this to a collector process to concatenate without knowledge of the rank
    output:
        path "rank_combined.pickle", emit: pickle
    script:
        """
        cva_rank_combine.py $results
        """
    stub:
        """
        touch "rank_combined.pickle"
        """
}

process publishAnalysis {
    label 'canBeLocal'
    input:
        path "rank_analysis.ipynb"
        path "rank_combined.pickle"
    output:
        path "rank_analysis.ipynb"
        path "rank_analysis.html"
        path "rank_analysis.tsv"
    publishDir { params.publish_dir }, mode: 'copy', overwrite: true
    script:
        """
        jupyter nbconvert --execute --to html rank_analysis.ipynb
        """
}

// Regularisation selection
process reguBiCv {
    // Perform bi-fold cross validation for a given rank and matrix
    // When multiple inputs, each input must come from a different channel
    // Here this is okay, as we will make a channel which emits the desired 
    // ranks to search.
    input:
        tuple val(rank), path('submats.npz'), val(iter)
    // The results file has all the params in it, so process which joins
    // can use that instead
    output:
        path('results.json')
    cpus 4
    script:
        """
        bicv_regu.py \
        --folds submats.npz \
        --seed $params.seed \
        --rank $rank \
        --num_runs 1 \
        --max_iter $params.nmf_max_iter \
        --shuffle_num $iter \
        --verbose \
        ${params.alpha.join(" ")}
        """
    stub:
        """
        touch results.json
        """
}

process reguCombineScores {
    label 'canBeLocal'
    input:
        path(scores, stageAs: "results*.json")
    // We will publish this as it is one of the primary outputs
    publishDir { params.publish_dir }, mode: 'copy', overwrite: true
    // The output file here will have the rank as a column, so we can just pass
    // this to a collector process to concatenate without knowledge of the rank
    output:
        path "biCV_regu_evar.json"
        path "biCV_regu_rss.json"
        path "biCV_regu_reco_error.json"
        path "biCV_regu_cosine.json"
        path "biCV_regu_l2norm.json"
        path "biCV_regu_sparsity.json"
    script:
        """
        bicv_regu_combine.py $scores
        """
    stub:
        """
        touch "biCV_regu_evar.json"
        touch "biCV_regu_rss.json"
        touch "biCV_regu_reco_error.json"
        touch "biCV_regu_cosine.json"
        touch "biCV_regu_l2norm.json"
        touch "biCV_regu_sparsity.json"
        """
}

process reguPublishAnalysis {
    label 'canBeLocal'
    input:
        path "regu_analysis.ipynb"
        path "biCV_evar.json"
        path "biCV_rss.json"
        path "biCV_reco_error.json"
        path "biCV_cosine.json"
        path "biCV_l2norm.json"
        path "biCV_sparsity.json"
    output:
        path "regu_analysis.ipynb"
        path "regu_analysis.html"
    publishDir { params.publish_dir }, mode: 'copy', overwrite: true
    script:
        """
        jupyter nbconvert --execute --to html regu_analysis.ipynb
        """
    stub:
        """
        touch regu_analysis.ipynb
        touch regu_analysis.html
        """
}

workflow bicv_regu {
    take:
    shuffles
    
    main:
    // CHANNELS
    // Ranks to search across
    rank_channel = Channel.from(params.regu_rank).view()
    // Alphas to search across
    alpha_channcel = Channel.from(params.alpha)

    // PROCESSES
    // Perform BiCV on each shuffle for each rank
    // Map to make sure that the second element of the tuple is a list of files,
    // rather than all being flattened into a single list
    bicv_res = reguBiCv(
        rank_channel
        .combine(shuffles)
    ).view( { it } )
    comb_res = reguCombineScores(bicv_res.collect())
    // Add the analysis notebook to the output
    reguPublishAnalysis(file("resources/bicv_regu_analysis.ipynb"), comb_res)
}

workflow bicv_rank {
    take:
    shuffles

    main:
    rank_channel = Channel.of(params.rankslist.get(0)..params.rankslist.get(1))

    // PROCESSES
    // Perform BiCV on each shuffle for each rank
    // Map to make sure that the second element of the tuple is a list of files,
    // rather than all being flattened into a single list
    bicv_res = rankBiCv(
        rank_channel
        .combine(shuffles.flatten())
    )

    // Combine results
    bicv_comb = rankCombineScores(bicv_res.collect())

    publishAnalysis(file("resources/cva_rank_analysis.ipynb"), bicv_comb)
    
}

workflow {
    // The input data - value channel so it can be consumed endlessly
    data_channel = file(params.matrix)
    // Shuffle and split matrix n times
    splits = biCVSplit(data_channel)

    // Rank selection
    bicv_rank(splits)
    // Regularisation selection
    // bicv_regu(splits)
}
