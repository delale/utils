rm(list = ls())

# working directory
setwd("C:/Users/adelu/Documents/UZH/Thesis/audio")

library(tidyverse)

# choose annotation file
fyle <- file.choose()

# annotation df
annot_dd <- read.delim(fyle, sep = "\t")

# file name
fyle_name <- str_split(fyle, "\\\\")[[1]][9]
str_sub(fyle_name, start = -4) <- ""

# metadata df
meta_dd <- read.csv("meta_test.csv")

colnames(annot_dd)[1] <- "Name"

# lifehistory info df
lh <- read.csv("C:/Users/adelu/Documents/UZH/Thesis/individual_selection/selection_dfs/sel_ind_info.csv")

# to add to metadata df
newdd <- data.frame(matrix(nrow = nrow(annot_dd), ncol = ncol(meta_dd)))
colnames(newdd) <- colnames(meta_dd)

# gather general info
rec_date <- readline(prompt = "rec date: ")
rec_date <- as.Date(rec_date, format = "%d/%m/%Y")
code <- str_split(fyle_name, "_")[[1]][1]
source_fyle <- paste(fyle_name, "WAV", sep = ".")
group_name <- readline(prompt = "Group Name: ")
bd <- lh %>%
    filter(Code == code) %>%
    select(BirthDate) %>%
    as.character() %>%
    as.Date()
age_d <- as.numeric(rec_date - bd)
age_y <- age_d / 365
id <- lh %>%
    filter(Code == code) %>%
    select(IndividID) %>%
    as.numeric()
sex <- lh %>%
    filter(Code == code) %>%
    select(Sex) %>%
    as.character()

# creates call file names
call_fyle <- c()

for (i in 1:nrow(annot_dd)) {
    if (i < 10) {
        tmp <- paste(fyle_name, "_0", as.character(i), ".WAV", sep = "")
    } else {
        tmp <- paste(fyle_name, "_", as.character(i), ".WAV", sep = "")
    }

    call_fyle <- append(call_fyle, tmp)
}

# gathers call type from annotation
call_type <- apply(annot_dd["Name"], 1, function(x) {
    str_sub(x, end = 2) <- ""
    return(x)
})

newdd["GroupName"] <- rep(group_name, nrow(newdd))
newdd["Sex"] <- rep(sex, nrow(newdd))
newdd["RecDate"] <- (rep(rec_date, nrow(newdd)))
newdd["Code"] <- rep(code, nrow(newdd))
newdd["ID"] <- rep(id, nrow(newdd))
newdd["AgeDays"] <- rep(age_d, nrow(newdd))
newdd["AgeYears"] <- rep(age_y, nrow(newdd))
newdd["SourceFile"] <- rep(source_fyle, nrow(newdd))
newdd["CallFile"] <- call_fyle
newdd["CallType"] <- call_type
newdd["CallTime"] <- annot_dd["Start"]

meta_dd <- rbind(meta_dd, newdd)
write.csv(meta_dd, "meta_test.csv", row.names = FALSE)