# clear the environment variables and plot
rm(list=ls())
dev.off()

# load necessary library
library(stringr)

# set working directory
setwd("~/Sem_4/DS/Assignment/A1")

# read data
df <- read.csv("marketbasketData.csv", perl = TRUE)
View(df)

# Create Basket
a <- list()
for(i in 1:nrow(df)){
  for (j in 1:length(df[i,])){
    if(df[i,j]==' true'){
      a <- c(a,names(df)[j])
    }
  }
  a <- c(a, sep="\n")
}

# Convert to string
l <- toString(a); l

# write intermediate output to csv
write.csv(as.data.frame(l),file="temp.csv", quote=F,row.names=T)

# read csv for pre-processing
file_lines <- readLines("temp.csv")
# remove preceeding commas
file_lines <- gsub(',+', ',', gsub('^,+|,+$', '', file_lines))
clean <- file_lines[-1]
# write clean data to csv file
writeLines(clean, "basket.csv")