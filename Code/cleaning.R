#--------------------------------------------------------------------
#Detecting Topic Merges and Splits in Dynamic Political Conversations
#--------------------------------------------------------------------
# Cláudia Oliveira
# Supervisor - Prof. Dr. Álvaro Figueira
# Faculty of Science, University of Porto
# -------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
# Libraries
# -------------------------------------------------------------------------------------------------------------------

library(readr)
library(dplyr)
library(stringr)

# -------------------------------------------------------------------------------------------------------------------
# Importing data set
# -------------------------------------------------------------------------------------------------------------------

df = read_csv("C:/Users/claud/OneDrive/Ambiente de Trabalho/tese/Code/datasets/tweets.csv")

str(df)
dim(df)
summary(df)

# -------------------------------------------------------------------------------------------------------------------
# Joining info from different columns
# -------------------------------------------------------------------------------------------------------------------

# Only English tweets
df <- df %>%
  filter(language == "en")

# Remove #
df <- df %>%
  mutate(
    hashtags_clean = ifelse(
      is.na(hashtags),
      "",
      str_replace_all(hashtags, "#", "") |> str_replace_all(",", "")
    )
  )

# Remove irrelevant info from urls
df <- df %>%
  mutate(
    url_words = ifelse(
      is.na(urls),
      "",
      urls |>
        str_replace_all("https?://", "") |>    # remove http:// or https://
        str_replace_all("www\\.", "") |>       # remove www.
        str_replace_all("[^a-zA-Z/]", " ") |>  # replace non-letters with space
        str_replace_all("/", " ") |>           # replace slashes with space
        str_squish()                           # trim extra spaces
    )
  )

irrelevant_words <- c("com", "org", "gov", "net", "html")

df$url_words <- sapply(str_split(df$url_words, " "), function(x) {
  paste(setdiff(x, irrelevant_words), collapse = " ")
})

# Concatenate the tweet with the cleaned hashtags and urls
df <- df %>%
  mutate(
    text_combined = str_squish(
      paste(text, hashtags_clean, url_words)
    )
  )

tweets = df$text_combined

# -------------------------------------------------------------------------------------------------------------------
# NLP
# -------------------------------------------------------------------------------------------------------------------

process_tweets <- function(tweets) {
  tweets <- replace_html(tweets)                             # expose ampersands (&) to be removed later
  tweets <- gsub("@\\w+", "", tweets)                        # remove usernames
  tweets <- gsub("\\.(\\S)", ". \\1", tweets, perl = TRUE)   # remove dots
  corpus <- corpus(tweets)
  toks <- tokens(
    corpus,
    remove_punct = TRUE,
    remove_symbols = TRUE
  )
  toks <- tokens_tolower(toks)
  toks <- tokens_remove(toks, stopwords("english"))
  toks <- tokens_remove(toks, pattern = "^\\d+$", valuetype = "regex")
  toks <- tokens_wordstem(toks)
  return(toks)
}









