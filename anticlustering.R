library(readxl) 
library(anticlust)  
library(writexl) 
library(ggplot2)  
start_time <- Sys.time()

file_path <- ".../mydatafinal_train.xlsx"
data_raw <- read_excel(file_path)

# Extract data with label 1 and perform anticlustering
data_label1 <- data_raw[data_raw[[1]] == 1, ]
anticlusters1 <- anticlustering(
  data_label1[, 2:(ncol(data_label1) - 1)],  # Keep feature columns
  objective = "diversity",
  method = "exchange",
  categories = data_label1[, c("binary1", "binary2", "binary3", "binary4")],
  K <- 3,
  standardize = TRUE,
  repetitions = 30
)
data_label1$Cluster <- anticlusters1  # Add cluster ID


# Extract data with label 2 and perform anticlustering
data_label2 <- data_raw[data_raw[[1]] == 2, ]
anticlusters2 <- anticlustering(
  data_label2[, 2:(ncol(data_label2) - 1)],  # Keep feature columns
  objective = "diversity",
  method = "exchange",
  categories = data_label2[, c("binary1", "binary2", "binary3", "binary4")],
  K <- 2,
  standardize = TRUE,
  repetitions = 30
)
data_label2$Cluster <- anticlusters2  # Add cluster ID


# Save results
write_xlsx(data_label1, ".../data_label1_with_final3-2.xlsx")
write_xlsx(data_label2, ".../data_label2_with_final3-2.xlsx")

end_time <- Sys.time()

print(paste("anticlustering time：", round(difftime(end_time, start_time, units = "secs"), 2), "seconds"))

data_label1 <- read_excel(".../data_label1_with_final3-2.xlsx")
data_label2 <- read_excel(".../data_label2_with_final3-2.xlsx")

# Extract data with label 3 from the original dataset
file_path <- ".../mydatafinal_train.xlsx"
data_raw <- read_excel(file_path)
data_label3 <- data_raw[data_raw[[1]] == 3, ]


# Select specified clusters
selected_cluster1 <- 1 # Cluster ID for label 1
selected_cluster2 <- 1 # Cluster ID for label 2

# Extract samples of the specified cluster from label 1 data
selected_data_label1 <- data_label1[data_label1$Cluster == selected_cluster1, ]

# Extract samples of the specified cluster from label 2 data
selected_data_label2 <- data_label2[data_label2$Cluster == selected_cluster2, ]

data_label3$Cluster <- NA



# Combine all data
final_combined_data <- rbind(
  selected_data_label1,
  selected_data_label2,
  data_label3
)

# Save final result
output_path <- ".../train32.xlsx"
write_xlsx(final_combined_data, output_path)

selected_clusters1m <- c(2, 3)   # Remaining clusters for label 1 (removed points)
selected_clusters2m <- c(2)    # Remaining clusters for label 2 (removed points)


selected_data_label1_multiple <- data_label1[data_label1$Cluster %in% selected_clusters1m, ]

selected_data_label2_multiple <- data_label2[data_label2$Cluster %in% selected_clusters2m, ]

final_combined_data_multiple <- rbind(
  selected_data_label1_multiple,
  selected_data_label2_multiple
)

# Save new combination: removed points as additional test set
output_path_multiple <- ".../addtest32.xlsx"
write_xlsx(final_combined_data_multiple, output_path_multiple)
