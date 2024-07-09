import java.util.stream.Collectors;

// Workaround for allowing comma separated list of rank upper/lower bounds
def rl = params.ranks?.split(',') as List
params.rankslist = rl.collect { it as int }

process biCVSplit {
    // Shuffle data matrix and split into 9 submatrices for bi-cross validation
    label 'largemem'
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
    label 'largemem'
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
    label 'largemem'
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

process publishRankDecompositions {
    input:
        path "rank_decomposition.ipynb"
        path data
        val rank
    output:
        path "decomposition.ipynb"
        path "decomposition.html"
        path("model", type: "dir") 
    publishDir { "${params.publish_dir}/${rank}" }, mode: 'copy', overwrite: true
    script:
        """
        papermill rank_decomposition.ipynb decomposition.ipynb \
        -p data ${data} -p rank ${rank} -p top_n 1 \
        -p seed ${params.seed} -p max_iter ${params.nmf_max_iter}
        jupyter nbconvert --to html decomposition.ipynb
        """
}

// Regularisation selection
process reguBiCv {
    // Perform bi-fold cross validation for a given rank and matrix
    // When multiple inputs, each input must come from a different channel
    // Here this is okay, as we will make a channel which emits the desired 
    // ranks to search.
    input:
        tuple val(rank), val(alpha), path(x)
    // The results file has all the params in it, so process which joins
    // can use that instead
    output:
        tuple val(rank), path('results.pickle')
    cpus 4
    script:
        """
        cva_alpha.py \
        --folds $x \
        --seed $params.seed \
        --rank $rank \
        --alpha $alpha \
        --max_iter $params.nmf_max_iter \
        --verbose 
        """
    stub:
        """
        touch results.json
        """
}

process reguCombineScores {
    label 'largemem'
    input:
        tuple val(rank), path(results, stageAs: "results*.pickle")
    output:
        tuple val(rank), path("regu_combined.pickle")
    script:
        """
        cva_rank_combine.py -g alpha $results
        """
    stub:
        """
        touch "regu_combined.pickle"
        """
}

process reguPublishAnalysis {
    label 'largemem'
    input:
        path "regu_analysis.ipynb"
        tuple val(rank), path("regu_combined.pickle")
    output:
        path "regu_analysis.ipynb"
        path "regu_analysis.html"
        path "regu_analysis.tsv"
    publishDir { "${params.publish_dir}/${rank}" }, mode: 'copy', overwrite: true
    script:
        """
        jupyter nbconvert --execute --to html regu_analysis.ipynb
        """
}

process reguPublishDecomposition {
    input:
        path matrix
        tuple val(rank), path(regu_res)
    output:
        path("regularised_model", type: "dir")
    publishDir { "${params.publish_dir}/${rank}" }, mode: 'copy', overwrite: true
    script:
        """
        cva_decompose.py \
        --input $matrix \
        --regu_res $regu_res \
        --seed $params.seed \
        --rank $rank \
        --max_iter $params.nmf_max_iter \
        --l1_ratio $params.l1_ratio \
        --random_starts $params.random_starts
        """
}

workflow bicv_regu {
    take:
    shuffles
    
    main:
    // CHANNELS
    // Ranks to search across
    rank_channel = Channel.from(params.regu_rank)
    // Alphas to search across
    alpha_channel = Channel.from(params.alpha)
    // Make a channel which is a combination of rank, alpha and shuffle
    regu_sel_channel = rank_channel
        .combine(alpha_channel)
        .combine(shuffles.flatten())

    // PROCESSES
    // Perform BiCV on each shuffle for each rank
    // Map to make sure that the second element of the tuple is a list of files,
    // rather than all being flattened into a single list
    bicv_res = reguBiCv(regu_sel_channel)
    // Group runs on the same rank together
    grpd_regu_res = bicv_res.groupTuple(
        by: 0,
        size: params.alpha.size()*params.n,
        remainder: true
    )
    grpd_regu_res.view()
    comb_res = reguCombineScores(grpd_regu_res)
    comb_res.view()
    // Add the analysis notebook to the output
    reguPublishAnalysis(file("resources/cva_regu_analysis.ipynb"), comb_res)
    // Make a regularised decomposition for each rank requested
    reguPublishDecomposition(file(params.matrix), comb_res)
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

    // Regularisation selection
    bicv_regu(splits)
    // Rank selection
    bicv_rank(splits)
    // Making decompositions for all the target ranks
    publishRankDecompositions(
        file("resources/rank_decomposition.ipynb"),
        data_channel,
        Channel.of(params.rankslist.get(0)..params.rankslist.get(1))
    )
}
