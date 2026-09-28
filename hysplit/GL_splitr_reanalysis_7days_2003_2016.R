rm(list = ls())

library(tidyverse)
# devtools::install_github("rich-iannone/splitr")
library(splitr)
library(sf)

# file path
path <- "~/Documents/Research/GeomNMF/hysplit"
met_dir <- file.path(path, "met")
dir.create(met_dir, recursive = TRUE, showWarnings = FALSE)

# read data
GL_df <- read_csv("~/Documents/Research/GeomNMF/data/GL_measurements.csv")
days_all <- sort(unique(GL_df$SampleDate))

# HYSPLIT
options(timeout = 600)

traj_3 <- list()
days_3 <- days_all[grepl("2003-", days_all)]
total_3 <- length(days_3)
for (i in 1:total_3) {
  d <- days_3[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_3, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  )
  traj_3[[d]] <- traj
}
save(traj_3, file = file.path(path, "GL_hysplit_reanalysis_7days_2003.RData"))

traj_4 <- list()
days_4 <- days_all[grepl("2004-12-", days_all)]
total_4 <- length(days_4)
for (i in 1:total_4) {
  d <- days_4[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_4, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  ) 
  traj_4[[d]] <- traj
}
save(traj_4, file = file.path(path, "GL_hysplit_reanalysis_7days_2004.RData")) 

traj_5 <- list()
days_5 <- days_all[grepl("2005-", days_all)]
total_5 <- length(days_5)
for (i in 1:total_5) {
  d <- days_5[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_5, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  ) 
  traj_5[[d]] <- traj
}
save(traj_5, file = file.path(path, "GL_hysplit_reanalysis_7days_2005.RData")) 

traj_6 <- list()
days_6 <- days_all[grepl("2006-", days_all)]
total_6 <- length(days_6)
for (i in 1:total_6) {
  d <- days_6[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_6, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  ) 
  traj_6[[d]] <- traj
}
save(traj_6, file = file.path(path, "GL_hysplit_reanalysis_7days_2006.RData")) 

traj_7 <- list()
days_7 <- days_all[grepl("2007-", days_all)]
total_7 <- length(days_7)
for (i in 1:total_7) {
  d <- days_7[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_7, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  ) 
  traj_7[[d]] <- traj
}
save(traj_7, file = file.path(path, "GL_hysplit_reanalysis_7days_2007.RData")) 

traj_9 <- list()
days_9 <- days_all[grepl("2009-", days_all)]
total_9 <- length(days_9)
for (i in 1:total_9) {
  d <- days_9[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_9, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  ) 
  traj_9[[d]] <- traj
}
save(traj_9, file = file.path(path, "GL_hysplit_reanalysis_7days_2009.RData")) 

traj_10 <- list()
days_10 <- days_all[grepl("2010-", days_all)]
total_10 <- length(days_10)
for (i in 1:total_10) {
  d <- days_10[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_10, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  ) 
  traj_10[[d]] <- traj
}
save(traj_10, file = file.path(path, "GL_hysplit_reanalysis_7days_2010.RData")) 

traj_11 <- list()
days_11 <- days_all[grepl("2011-", days_all)]
total_11 <- length(days_11)
for (i in 1:total_11) {
  d <- days_11[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_11, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  )
  traj_11[[d]] <- traj
}
save(traj_11, file = file.path(path, "GL_hysplit_reanalysis_7days_2011.RData"))

traj_12 <- list()
days_12 <- days_all[grepl("2012-", days_all)]
total_12 <- length(days_12)
for (i in c(1:6, 8:total_12)) {
  d <- days_12[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_12, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  )
  traj_12[[d]] <- traj
}
save(traj_12, file = file.path(path, "GL_hysplit_reanalysis_7days_2012.RData"))

traj_13 <- list()
days_13 <- days_all[grepl("2013-", days_all)]
total_13 <- length(days_13)
for (i in 1:total_13) {
  d <- days_13[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_13, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  )
  traj_13[[d]] <- traj
}
save(traj_13, file = file.path(path, "GL_hysplit_reanalysis_7days_2013.RData"))

traj_15 <- list()
days_15 <- days_all[grepl("2015-", days_all)]
total_15 <- length(days_15)
for (i in 1:total_15) {
  d <- days_15[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_15, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  )
  traj_15[[d]] <- traj
}
save(traj_15, file = file.path(path, "GL_hysplit_reanalysis_7days_2015.RData"))

traj_16 <- list()
days_16 <- days_all[grepl("2016-", days_all)]
total_16 <- length(days_16)
for (i in 1:total_16) {
  d <- days_16[i]
  cat(sprintf("[%d/%d] Processing %s...\n", i, total_16, d))
  traj <- hysplit_trajectory(
    lat = 72.58,
    lon = -38.46,
    height = c(20, 50, 100, 650), # meters
    duration = 168,
    days = d,
    direction = "backward",
    daily_hours = c(0, 6, 12, 18),
    met_type = "reanalysis",
    extended_met = TRUE,       # <-- this enables MLH and other met variables
    met_dir = met_dir
  )
  traj_16[[d]] <- traj
}
save(traj_16, file = file.path(path, "GL_hysplit_reanalysis_7days_2016.RData"))

# mixed layer height: https://claude.ai/share/4f04e04f-ddcf-4cce-b388-9eec44a00bf1
