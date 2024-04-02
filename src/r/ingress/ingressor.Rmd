# Data Ingressor

## load libraries:
```{R}
source("src/r/utils.R")

install_or_load(c("ggplot2", "dplyr", "tidyr"))
```


## Create a class to load data and split it into train and test sets:
```{R}
setClass("Ingressor", slots = c(
  file_path = "character",
  #data = "data.frame",
  )
)

setMethod(
    "run",
    signature("Ingressor"),
    function(object){
        object@data = read.csv(object@file_path)
        return(object)
    }
)

ingressor <- new("Ingressor", file_path = "artifacts/data/iris.csv")
```