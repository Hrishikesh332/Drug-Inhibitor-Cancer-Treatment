library("biomaRt")
target_gene <- read.csv(args[1], header = FALSE)
colnames(target_gene) <- c('external_gene_name')
ensembl_human = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
genes_name_list <- target_gene$external_gene_name
result <- getBM(attributes = c('external_gene_name', 'entrezgene')
                  , filters = c('external_gene_name'),
                  values = genes_name_list, mart = ensembl_human)

result <- merge(result, target_gene, by = 'external_gene_name')
dup<- duplicated(result$external_gene_name) | duplicated(result$external_gene_name, fromLast = TRUE)

unique_found <- result[!dup,]
dup_found <- result[dup,]

write.csv(unique_found, file = "found_genes.csv")
write.csv(dup_found, file = "found_duplicates.csv")