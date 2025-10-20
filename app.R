# This is a minimal R shiny app
# Users can ask questions/query the irAE dataset from FAERS that David formatted
# Backend code in python
# Connecting to shiny frontend using reticulate package

library(reticulate)
library(shiny)

# Set working directory to github repo with app.R script
setwd("~/Documents/Tan_Lab/Projects/Python_irAE_LLM_Query/")

# Ensure Python environment
use_condaenv("irae_agent", required = TRUE)

# Import Python manager
manager_module <- import_from_path("manager", path = "src")

# Read cleaned irae table into memory 
df <- r_to_py(read.csv("data/irae_data_cleaned.csv", stringsAsFactors = FALSE))

# Instantiate manager class
m <- manager_module$Manager(df)

# Simple page with sidebar to ask question.
# Also has a main panel which will be populated with the results of the question
# Will export python code from the model as well as the plot/numeric/tabular output

ui <- fluidPage(
  titlePanel("irAE Dataset LLM Assistant"),
  
  sidebarLayout(
    sidebarPanel(
      textInput("question", "Ask a question:", ""),
      actionButton("run", "Submit")
    ),
    
    mainPanel(
      h4("Generated Python code"),
      verbatimTextOutput("code"),
      hr(),
      h4("Result"),
      uiOutput("resultUI")
    )
  )
)

# --- Server ---
server <- function(input, output, session) {
  
  # single source of truth for the last Python result (a Py dict-like)
  resRV <- shiny::reactiveVal(NULL)
  
  shiny::observeEvent(input$run, {
    shiny::req(input$question)
    # call Python
    pyres <- m$process_question(input$question)
    # Force convert the Python dictionary-like object to an R list
    res <- reticulate::py_to_r(pyres)
    resRV(res)
  })
  
  # code block
  output$code <- shiny::renderText({
    res <- resRV()
    if (is.null(res)) return("No result yet.")
    code <- res$code
    if (is.null(code)) "No code generated." else as.character(code)
  })
  
  # table renderer (used only when result type == "dataframe")
  output$dataTable <- shiny::renderTable({
    res <- resRV()
    if (is.null(res)) return(NULL)
    rdata <- tryCatch(py_to_r(res$data), error = function(e) NULL)
    if (is.data.frame(rdata)) {
      rdata[rdata == ""] <- NA
      head(rdata, 25)
      } else {
        NULL
      }
  })
  
  # text renderer (used for "text", "number", "error", and non-df query results)
  output$textResult <- shiny::renderText({
    res <- resRV()
    if (is.null(res)) return(NULL)
    as.character(res$data)
  })
  
  # dynamic UI decides which placeholder to show; renderers are defined above
  output$resultUI <- shiny::renderUI({
    res <- resRV()
    if (is.null(res)) return(NULL)
    
    res_type <- res$type
    res_data <- res$data
    
    switch(
      res_type,
      "plot" = shiny::tags$img(
        src = paste0("data:image/png;base64,", res_data),
        style = "max-width:100%; height:auto;"
      ),
      "number" = shiny::div(
        style = "font-size:2em; font-weight:bold; color:#2C3E50;",
        as.character(res_data)
      ),
      "dataframe" = shiny::tableOutput("dataTable"),
      "text" = shiny::verbatimTextOutput("textResult"),
      "error" = shiny::div(
        style = "color:red; white-space:pre-wrap;",
        as.character(res_data)
      ),
      shiny::div("No result to display.")
    )
  })
}

shiny::shinyApp(ui, server)

