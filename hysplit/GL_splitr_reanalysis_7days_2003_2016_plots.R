rm(list = ls())

library(tidyverse)
theme_set(theme_bw())
library(sf)
# devtools::install_github("rich-iannone/splitr")
library(splitr)
library(terra) # provides necessary GIS tools
library(ggmap) # used to overlay trajectories onto maps
library(patchwork)

# file path
path <- "~/Documents/Research/GeomNMF/hysplit"
met_dir <- file.path(path, "met")

# source code
laea <- "+proj=laea +lat_0=90 +lon_0=0 +datum=WGS84"
to_sf <- function(df, x, y) {
  x <- rlang::as_name(rlang::ensym(x))
  y <- rlang::as_name(rlang::ensym(y))
  df %>%
    sf::st_as_sf(coords = c(x, y), crs = '+proj=longlat +datum=WGS84') %>%
    sf::st_transform(crs = laea)  # N. Pole
}

#==============#
# hysplit data #
#==============#
years <- c(2003:2007, 2009:2013, 2015:2016)
for (year in years) {
  filename <- paste0("GL_hysplit_reanalysis_7days_", year, ".RData")
  load(file.path(path, filename))
}
traj_long <- bind_rows(traj_3, traj_4, traj_5, traj_6, traj_7, traj_9, traj_10, 
                       traj_11, traj_12, traj_13, traj_15, traj_16) %>% 
  na.omit() %>% 
  mutate(days = date(traj_dt_i), 
         daily_hours = hour(traj_dt_i), 
         umlh = (height <= mixdepth))

#===========#
# Load data #
#===========#
Y_df <- read_csv(paste0("~/Documents/Research/GeomNMF/data/GL_measurements.csv"))
Y_df$SampleDate %>% range()
Y_df %>% 
  mutate(month = month(SampleDate, label = T)) %>% 
  group_by(month) %>% 
  summarize(n = n())
plt_anthro <- Y_df %>%
  pivot_longer(c("Cu", "Ni", "Pb", "S", "Zn"), names_to = "pollutant", values_to = "value") %>% 
  group_by(pollutant) %>% 
  complete(SampleDate = seq(min(SampleDate), max(SampleDate), by = "day")) %>%
  ungroup() %>% 
  ggplot() + 
  geom_line(aes(SampleDate, value)) +
  facet_wrap(~pollutant, scales = "free_y", nrow = 6) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") + 
  labs(x = "Date", y = "Measurement (ng/m3)")
# ggsave(filename = paste0("~/Documents/Research/GeomNMF/results/GL/figure/GL_anthropogenic_elements_time_series.pdf"),
#        plt_anthro, height = 6, width = 8)

plt_natural <- Y_df %>%
  pivot_longer(c("Al", "Ca", "Cl", "Si", "Ti"), names_to = "pollutant", values_to = "value") %>% 
  group_by(pollutant) %>% 
  complete(SampleDate = seq(min(SampleDate), max(SampleDate), by = "day")) %>%
  ungroup() %>% 
  ggplot() + 
  geom_line(aes(SampleDate, value)) +
  facet_wrap(~pollutant, scales = "free_y", nrow = 6) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") + 
  labs(x = "Date", y = "Measurement (ng/m3)")
# ggsave(filename = paste0("~/Documents/Research/GeomNMF/results/GL/figure/GL_natural_elements_time_series.pdf"),
#        plt_natural, height = 6, width = 8)

#=================================#
# Load estimated source intensity #
#=================================#
K <- 6
min_K <- 10*K
W_df <- read_csv(paste0("~/Documents/Research/GeomNMF/results/GL/GL_scaled_K", K, "_minK", min_K, "_W_tilde_hat.csv"))
source_cols <- paste("Source", 1:K)
colnames(W_df)[1:K] <- source_cols
W_long <- W_df %>%
  pivot_longer(starts_with("Source"), names_to = "source", values_to = "intensity") 

#======# 
# Plot #
#======#
# Greenland summit 
summit_sf <- tibble(lat = 72.58,
                    lon = -38.46) %>% 
  to_sf(lon, lat)

# world basemap
world <- rnaturalearth::ne_countries(scale = "medium", returnclass = "sf")
world_proj <- sf::st_transform(world, crs = laea)

# define Arctic extent 
arctic_extent <- 6e6  # ~6000 km radius from pole

# convert lon lat to projected coordinates 
xy <- traj_long %>%
  select(lon, lat) %>%
  to_sf(lon, lat) %>%
  sf::st_coordinates()

# blank grid with 100 km  
g <- rast(ext(-arctic_extent, arctic_extent, -arctic_extent, arctic_extent), 
          resolution = 1e5, crs = laea)

# attach grid cell index
traj_cell <- traj_long %>%
  mutate(cell_id = cellFromXY(g, xy), # integer index of the cell in g that contains each xy
         run_id = paste(traj_dt_i, height_i)) # date, time, height at Summit (slightly less than 16 x 1861 due to missing comb)
traj_cell %>% count(run_id) %>% count(n) # expect ~169

# number of endpoints (residence time) from trajectories starting on `days` at Summit in each `cell_id` 
tau_df <- traj_cell %>%
  filter(!is.na(cell_id)) %>% # drop endpoints falling outside the grid
  count(cell_id, days, name = "tau")

# potential source contribution function trajectory
q <- 0.99
thr <- W_long %>%
  rename(days = SampleDate) %>%
  group_by(source) %>%
  summarize(cutoff = quantile(intensity, q, na.rm = TRUE), .groups = "drop")
pscf_df <- tau_df %>%
  inner_join(W_long %>% rename(days = SampleDate), by = "days",
             relationship = "many-to-many") %>%
  inner_join(thr, by = "source") %>%
  group_by(source, cell_id) %>%
  summarize(m = sum(tau * (intensity > cutoff)),   # polluted endpoints
            n = sum(tau),                           # all endpoints
            pscf = m / n,
            .groups = "drop")

# downweight tiny n cells 
summit_cell <- cellFromXY(g, sf::st_coordinates(summit_sf))

pscf_df$n %>% summary() # mean is way above median 
hist(pscf_df$n) # very skewed
nbar <- mean(pscf_df$n)
nmin <- median(pscf_df$n) # reliability cell count
pscf_df2 <- pscf_df %>%
  # mutate(w = case_when(n > 3.0*nbar ~ 1.00, # Polissar et al. (2001)
  #                      n > 1.5*nbar ~ 0.70,
  #                      n > 1.0*nbar ~ 0.42,
  #                      TRUE         ~ 0.05),
  #        pscf_w = pscf * w) %>%
  mutate(w = pmin(1, n / nmin), # <n_min ramps down, ≥n_min stays 1
         pscf_w = pscf * w) %>%
  filter(cell_id != summit_cell)

# rasterize
make_raster <- function(df, src, weight_col) {
  r <- g
  values(r) <- NA       # reuse g's geometry, blank it
  sub <- df %>% filter(source == src)
  r[sub$cell_id] <- sub[[weight_col]] # fill by cell index; unfilled stay NA; possible because r and sub has the same index
  names(r) <- src
  r
}

r_all <- rast(lapply(source_cols, function(s) make_raster(pscf_df2, s, "pscf_w")))
df_all <- as.data.frame(r_all, xy = TRUE, na.rm = TRUE) %>%
  tidyr::pivot_longer(-c(x, y), names_to = "source", values_to = "pscf") 

# add extra info 
# # add potential smelter sources
# ## Ni
# # Nikel (Pechenganickel)	69.41	30.23
# # Norilsk (Nadezhda)	69.35	88.20
# # Harjavalta (Boliden)	61.31	22.14
# nikel_lat <- 69.4; nikel_lon <- 30.2
# norilsk_lat <- 69.4; norilsk_lon <- 88.2
# fin_lat <- 61.3; fin_lon <- 22.1
# 
# ## Pb
# # Rönnskär (Boliden)	64.67	21.23
# swe_lat <- 64.7; swe_lon <- 21.2
# 
# s2_source_sf <- bind_rows(
#   to_sf(tibble(x = nikel_lon,   y = nikel_lat), x, y)   %>% mutate(site = "Nikel (Pechenganickel)"),
#   to_sf(tibble(x = norilsk_lon, y = norilsk_lat), x, y) %>% mutate(site = "Norilsk (Nadezhda)"),
#   to_sf(tibble(x = fin_lon,     y = fin_lat), x, y)     %>% mutate(site = "Harjavalta (Boliden)"),
#   to_sf(tibble(x = swe_lon,     y = swe_lat), x, y)     %>% mutate(site = "Rönnskär (Boliden)")
# ) %>% 
#   mutate(source = "Source 2")
# 
# # add Canadian Shield for Source 6
# # NRCan Physiographic Regions of Canada (FGDB)
# url <- "https://ftp.maps.canada.ca/pub/nrcan_rncan/Geology_Geologie/physiographic_regions_physiographiques/phys_reg.gdb.zip"
# download.file(url, "phys_reg.gdb.zip", mode = "wb")
# unzip("phys_reg.gdb.zip", exdir = "phys_reg")
# gdb <- list.files("phys_reg", pattern = "\\.gdb$", full.names = TRUE)[1]
# st_layers(gdb)
# reg <- st_read(gdb, layer = "Regions")
# names(reg)
# 
# # filter to the Shield
# shield <- reg[grepl("Shield", reg$RegionEn, ignore.case = TRUE), ]
# shield_df <- shield %>%
#   st_as_sf() %>%              # sfc -> sf, so mutate has columns to write to
#   st_transform(laea) %>%
#   mutate(source = "Source 6")
# 
# # add Calgary-Edmonton Corridor
# url <- "https://geo.statcan.gc.ca/geo_wa/rest/services/2021/Cartographic_boundary_files/MapServer/4/query"
# resp <- httr::GET(url, query = list(
#   where          = "CDUID IN ('4806','4808','4811')",
#   outFields      = "CDUID,CDNAME,CDTYPE,PRUID,LANDAREA,DGUID",
#   returnGeometry = "true",
#   outSR          = "4326",      # request WGS84 lon/lat
#   f              = "json"
# ))
# httr::stop_for_status(resp)
# 
# ce <- st_read(httr::content(resp, as = "text", encoding = "UTF-8"), quiet = TRUE)
# ce_df <- ce %>%
#   st_as_sf() %>%              # sfc -> sf, so mutate has columns to write to
#   st_transform(laea) %>%
#   mutate(source = "Source 1")
# 
# # add Arctic shipping routes 
# routes <- sf::st_read(file.path(path, "arctic_shipping_routes_3413.gpkg"), "routes")
# routes_df <- routes %>%
#   st_as_sf() %>%              # sfc -> sf, so mutate has columns to write to
#   st_transform(laea) %>%
#   mutate(source = "Source 3")

# save(s2_source_sf, # smelters
#      shield_df, # Canadian Shield
#      ce_df, # Calgary-Edmonton Corridor
#      routes_df, # Arctic shipping routes 
#      file = paste0(path, "/GL_hysplit_potential_sources.RData"))
load(file = paste0(path, "/GL_hysplit_potential_sources.RData"))

pscf99_w_extra_plt_hor <- ggplot() +
  geom_sf(data = world_proj, fill = "white", color = "gray60", linewidth = 0.2) +
  geom_sf(data = ce_df, aes(fill = "Calgary-Edmonton Corridor"),
          color = "#5e4b8b", linewidth = 0.7) +
  geom_sf(data = shield_df, aes(fill = "Canadian Shield"),
          color = "#b8a361", linewidth = 0.3, alpha = 0.5) +
  geom_point(data = df_all %>% filter(pscf > 0),
             aes(x, y, color = pscf), size = 0.5) +
  geom_sf(data = world_proj, fill = NA, color = "gray60", linewidth = 0.2) +
  geom_sf(data = summit_sf, color = "black", size = 2) +
  geom_sf(data = summit_sf, color = "red",   size = 1.5) +
  geom_sf(data = s2_source_sf, aes(shape = site), fill = "grey", color = "black", 
          size = 2, stroke = 0.6) +
  geom_sf(data = routes_df, aes(linetype = route),
          color = "grey15", linewidth = 0.7, alpha = 0.8) +
  facet_wrap(~ source, nrow = 2) +
  scale_color_distiller(name = "PSCF", palette = "Blues", direction = 1,
                        trans = "sqrt") + 
  scale_shape_manual(name   = "Smelter",
                     values = c("Nikel (Pechenganickel)" = 22, 
                                "Norilsk (Nadezhda)" = 24, 
                                "Harjavalta (Boliden)" = 23, 
                                "Rönnskär (Boliden)" = 25)) +
  scale_fill_manual(name = "Region",
                    values = c("Calgary-Edmonton Corridor" = "#5e4b8b",
                               "Canadian Shield"           = "#e6d8b5")) +
  scale_linetype_manual(name = "Shipping route",
                        values = c("Northern Sea Route"   = "longdash",
                                   "Northwest Passage"    = "solid",
                                   "Transpolar Sea Route" = "dotdash")) +
  coord_sf(xlim = c(-arctic_extent*0.9, arctic_extent*0.7),
           ylim = c(-arctic_extent, arctic_extent*0.8),
           crs = laea) +
  guides(color = guide_colorbar(order = 1),   # PSCF
         shape = guide_legend(order = 2),     # Smelter
         linetype = guide_legend(order = 3),  # Routes
         fill = guide_legend(order = 4)) +    # Region
  labs(x = "", y = "") + 
  theme(axis.text = element_blank(),
        axis.ticks = element_blank())
# ggsave("GL_hysplit_reanalysis_7days_2003_2016_pscf99_w_extra_bysource_horizontal.png", path = file.path(path, "figure"),
#        plot = pscf99_w_extra_plt_hor, width = 8, height = 5.5, dpi = 300)

# forward trajectories 
# Source 4: Saharan and Gobi 
dates_high <- W_df %>% 
  rename(src = "Source 4") %>% 
  filter(src > thr$cutoff[thr$source == "Source 4"]) %>% 
  pull(SampleDate)
start_dates_sahara <- dates_high - 7
start_dates_gobi <- dates_high - 14

# Sahara (central)
sahara_lat <- 23.0; sahara_lon <- 13.0
sahara_sf <- to_sf(tibble(x = sahara_lon, y = sahara_lat), x, y)

# Gobi desert
gobi_lat <- 42.0; gobi_lon <- 105.0
gobi_sf <- to_sf(tibble(x = gobi_lon, y = gobi_lat), x, y)

# traj_gobi <- list()
# total_dates_gobi <- length(start_dates_gobi)
# for (i in 1:total_dates_gobi) {
#   d <- start_dates_gobi[i]
#   cat(sprintf("[%d/%d] Processing %s...\n", i, total_dates_gobi, d))
#   traj <- splitr::hysplit_trajectory(
#     lat = gobi_lat,
#     lon = gobi_lon,
#     height = c(20, 50, 100, 650), # meters above ground level
#     duration = 168,
#     days = d,
#     direction = "forward",
#     daily_hours = c(0, 6, 12, 18),
#     met_type = "reanalysis",
#     extended_met = TRUE,       # <-- this enables MLH and other met variables
#     met_dir = met_dir
#   )
#   traj_gobi[[d]] <- traj
# }
# save(traj_gobi, file = file.path(path, "GL_hysplit_reanalysis_7days_from_Gobi.RData"))
# 
# traj_sahara <- list()
# total_dates_sahara <- length(start_dates_sahara)
# for (i in 1:total_dates_sahara) {
#   d <- start_dates_sahara[i]
#   cat(sprintf("[%d/%d] Processing %s...\n", i, total_dates_sahara, d))
#   traj <- splitr::hysplit_trajectory(
#     lat = sahara_lat,
#     lon = sahara_lon,
#     height = c(20, 50, 100, 650), # meters
#     duration = 168,
#     days = d,
#     direction = "forward",
#     daily_hours = c(0, 6, 12, 18),
#     met_type = "reanalysis",
#     extended_met = TRUE,       # <-- this enables MLH and other met variables
#     met_dir = met_dir
#   )
#   traj_sahara[[d]] <- traj
# }
# save(traj_sahara, file = file.path(path, "GL_hysplit_reanalysis_7days_from_Sahara.RData"))

load(file = file.path(path, "GL_hysplit_reanalysis_7days_from_Gobi.RData"))
traj_gobi <- bind_rows(traj_gobi)
load(file = file.path(path, "GL_hysplit_reanalysis_7days_from_Sahara.RData"))
traj_sahara <- bind_rows(traj_sahara)

# convert lon lat to projected coordinates 
xy_gobi <- traj_gobi %>%
  select(lon, lat) %>%
  to_sf(lon, lat) %>%
  sf::st_coordinates()
xy_sahara <- traj_sahara %>%
  select(lon, lat) %>%
  to_sf(lon, lat) %>%
  sf::st_coordinates()

# attach grid cell index
traj_gobi_cell <- traj_gobi %>%
  mutate(days = date(traj_dt_i),
         cell_id = cellFromXY(g, xy_gobi)) 
arctic_extent_v2 <- 8.5e6
g_sahara <- rast(ext(-arctic_extent_v2, arctic_extent_v2, -arctic_extent_v2, arctic_extent_v2), 
                 resolution = 1e5, crs = laea)
traj_sahara_cell <- traj_sahara %>%
  mutate(days = date(traj_dt_i),
         cell_id = cellFromXY(g_sahara, xy_sahara))

# number of endpoints (residence time) from trajectories starting on `days` at Gobi in each `cell_id` 
tau_gobi_df <- traj_gobi_cell %>%
  filter(!is.na(cell_id)) %>% # drop endpoints falling outside the grid
  count(cell_id, days, name = "tau")
tau_sahara_df <- traj_sahara_cell %>%
  filter(!is.na(cell_id)) %>% # drop endpoints falling outside the grid
  count(cell_id, days, name = "tau")

# normalized residence time 
tau_norm_gobi_df <- tau_gobi_df %>%
  inner_join(W_long %>% filter(source == "Source 4") %>% rename(days = SampleDate), by = "days") %>%
  group_by(cell_id) %>%
  summarize(tau2 = sum(tau), .groups = "drop") %>%
  mutate(tau_norm = tau2 / sum(tau2))          # fraction of total endpoint-time
tau_norm_sahara_df <- tau_sahara_df %>%
  inner_join(W_long %>% filter(source == "Source 4") %>% rename(days = SampleDate), by = "days") %>%
  group_by(cell_id) %>%
  summarize(tau2 = sum(tau), .groups = "drop") %>%
  mutate(tau_norm = tau2 / sum(tau2))          # fraction of total endpoint-time

# make raster step
r_gobi_all <- g              # copy the template grid
values(r_gobi_all) <- NA
r_gobi_all[tau_norm_gobi_df$cell_id] <- tau_norm_gobi_df$tau_norm
names(r_gobi_all) <- "tau_norm"
df_gobi_all <- as.data.frame(r_gobi_all, xy = TRUE, na.rm = TRUE) 

r_sahara_all <- g_sahara              # copy the template grid
values(r_sahara_all) <- NA
r_sahara_all[tau_norm_sahara_df$cell_id] <- tau_norm_sahara_df$tau_norm
names(r_sahara_all) <- "tau_norm"
df_sahara_all <- as.data.frame(r_sahara_all, xy = TRUE, na.rm = TRUE) 

s4_gobi_plt <- ggplot() +
  geom_sf(data = world_proj, fill = "white", color = "gray60", linewidth = 0.2) +
  geom_point(data = df_all %>% filter(source == "Source 4", pscf > 0),
             aes(x, y, color = pscf), size = 0.5) +
  geom_raster(data = df_gobi_all, aes(x, y, fill = tau_norm), alpha = 0.8) +
  geom_sf(data = world_proj, fill = NA, color = "gray60", linewidth = 0.2) +
  geom_sf(data = summit_sf, color = "black", size = 2) +
  geom_sf(data = summit_sf, color = "red", size = 1.5) +
  geom_sf(data = gobi_sf, color = "black", size = 2, shape = 15) +
  geom_sf(data = gobi_sf, color = "gray", size = 1.5, shape = 15) +
  facet_wrap(~ source, nrow = 2) +
  scale_color_distiller(name = "PSCF", palette = "Blues", direction = 1,
                        trans = "sqrt") +
  scale_fill_distiller(name = "Residence\ntime", palette = "Oranges", 
                        direction = 1, trans = "sqrt", 
                        labels = scales::label_percent(accuracy = 0.1)) +
  coord_sf(xlim = c(-arctic_extent*0.8, arctic_extent),
           ylim = c(-arctic_extent, arctic_extent),
           crs = laea) +
  labs(x = "", y = "") + 
  theme(axis.text = element_blank(),
        axis.ticks = element_blank()) +
  guides(color = guide_colorbar(order = 1))  
# ggsave("GL_hysplit_reanalysis_7days_2003_2016_Source4_Gobi.png", path = file.path(path, "figure"),
#        plot = s4_gobi_plt, width = 4.5, height = 4, dpi = 300)

s4_sahara_plt <- ggplot() +
  geom_sf(data = world_proj, fill = "white", color = "gray60", linewidth = 0.2) +
  geom_point(data = df_all %>% filter(source == "Source 4", pscf > 0),
             aes(x, y, color = pscf), size = 0.5) +
  geom_tile(data = df_sahara_all, aes(x, y, fill = tau_norm), alpha = 0.8) +
  geom_sf(data = world_proj, fill = NA, color = "gray60", linewidth = 0.2) +
  geom_sf(data = summit_sf, color = "black", size = 2) +
  geom_sf(data = summit_sf, color = "red", size = 1.5) +
  geom_sf(data = sahara_sf, color = "black", size = 2, shape = 18) +
  geom_sf(data = sahara_sf, color = "gray", size = 1.5, shape = 18) +
  facet_wrap(~ source, nrow = 2) +
  scale_color_distiller(name = "PSCF", palette = "Blues", direction = 1,
                        trans = "sqrt") +
  scale_fill_distiller(name = "Residence\ntime", palette = "Oranges", 
                       direction = 1, trans = "sqrt", 
                       labels = scales::label_percent(accuracy = 0.1)) +
  coord_sf(xlim = c(-arctic_extent*0.8, arctic_extent),
           ylim = c(-arctic_extent_v2, arctic_extent*0.7),
           crs = laea) +
  labs(x = "", y = "") + 
  theme(axis.text = element_blank(),
        axis.ticks = element_blank()) +
  guides(color = guide_colorbar(order = 1))    
# ggsave("GL_hysplit_reanalysis_7days_2003_2016_Source4_Sahara.png", path = file.path(path, "figure"),
#        plot = s4_sahara_plt, width = 4.5, height = 4, dpi = 300)
