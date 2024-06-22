import pandas as pd
import json
import plotly.express as px
import plotly.io as pio
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

# Step 1: Verify the file path and read the JSON file
file_path = 'result.json'

def read_large_json(file_path, chunk_size=1000):
    try:
        # Load the JSON file in chunks to handle large files
        with open(file_path, 'r') as file:
            data = json.load(file)
            # This code assumes the JSON file is an array of records
            if isinstance(data, list) and len(data) > chunk_size:
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size]
                    yield pd.DataFrame(chunk)
            else:
                yield pd.DataFrame(data)
    except FileNotFoundError:
        print(f"File {file_path} does not exist")
    except json.JSONDecodeError:
        print("Failed to decode JSON")

# Step 2: Process and visualize the data
data_frames = list(read_large_json(file_path))
if data_frames:
    data = pd.concat(data_frames, ignore_index=True)

    # Extract relevant data from 'messages' column
    messages_data = pd.json_normalize(data['messages'])

    # Convert 'date' to datetime for proper plotting
    messages_data['date'] = pd.to_datetime(messages_data['date'])

    # Step 3: Explore the Data
    print(messages_data.head())
    print(messages_data.info())

    # Step 4: Visualize the Data using Plotly
    # 1. Number of messages over time
    messages_data['date_only'] = messages_data['date'].dt.date
    messages_count = messages_data.groupby('date_only').size().reset_index(name='count')
    fig1 = px.line(messages_count, x='date_only', y='count', title='Number of messages over time')

    # 2. Distribution of message types
    message_types = messages_data['type'].value_counts().reset_index()
    message_types.columns = ['type', 'count']
    fig2 = px.pie(message_types, values='count', names='type', title='Distribution of message types')

    # 3. Top actors by message count
    top_actors = messages_data['from'].value_counts().head(10).reset_index()
    top_actors.columns = ['actor', 'count']
    fig3 = px.bar(top_actors, x='actor', y='count', title='Top actors by message count')

    # 4. Messages edited over time
    edited_messages = messages_data.dropna(subset=['edited'])
    edited_messages['edited_date_only'] = pd.to_datetime(edited_messages['edited']).dt.date
    edited_count = edited_messages.groupby('edited_date_only').size().reset_index(name='count')
    fig4 = px.line(edited_count, x='edited_date_only', y='count', title='Messages edited over time')

    # 5. Media types distribution
    media_types = messages_data['media_type'].value_counts().reset_index()
    media_types.columns = ['media_type', 'count']
    fig5 = px.bar(media_types, x='media_type', y='count', title='Distribution of media types')

    # 6. Number of messages per actor
    messages_per_actor = messages_data['from'].value_counts().reset_index()
    messages_per_actor.columns = ['actor', 'count']
    fig6 = px.bar(messages_per_actor, x='actor', y='count', title='Number of messages per actor')

    # 7. Top media types by count
    top_media_types = messages_data['media_type'].value_counts().head(10).reset_index()
    top_media_types.columns = ['media_type', 'count']
    fig7 = px.bar(top_media_types, x='media_type', y='count', title='Top media types by count')

    # 8. Messages sent by time of day
    messages_data['hour'] = messages_data['date'].dt.hour
    messages_by_hour = messages_data.groupby('hour').size().reset_index(name='count')
    fig8 = px.bar(messages_by_hour, x='hour', y='count', title='Messages sent by time of day')

    # 9. Length of messages over time
    messages_data['text_length'] = messages_data['text'].apply(lambda x: len(str(x)))
    avg_length_over_time = messages_data.groupby('date_only')['text_length'].mean().reset_index(name='avg_length')
    fig9 = px.line(avg_length_over_time, x='date_only', y='avg_length', title='Average length of messages over time')

    # 10. Top 10 most common words in messages
    stop_words = set(stopwords.words('english'))
    messages_data['clean_text'] = messages_data['text'].apply(lambda x: ''.join([char for char in str(x) if char not in string.punctuation]))
    messages_data['clean_text'] = messages_data['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    all_words = ' '.join(messages_data['clean_text'].tolist()).split()
    common_words = Counter(all_words).most_common(10)
    common_words_df = pd.DataFrame(common_words, columns=['word', 'count'])
    fig10 = px.bar(common_words_df, x='word', y='count', title='Top 10 most common words in messages')

    # 11. Reply patterns
    reply_count = messages_data.dropna(subset=['reply_to_message_id']).groupby('date_only').size().reset_index(name='count')
    fig11 = px.line(reply_count, x='date_only', y='count', title='Number of replies over time')

    # 12. Forwarded messages
    forwarded_count = messages_data.dropna(subset=['forwarded_from']).groupby('date_only').size().reset_index(name='count')
    fig12 = px.line(forwarded_count, x='date_only', y='count', title='Number of forwarded messages over time')

    # # Show the plots
    # fig1.show()
    # fig2.show()
    # fig3.show()
    # fig4.show()
    # fig5.show()
    # fig6.show()
    # fig7.show()
    # fig8.show()
    # fig9.show()
    # fig10.show()
    # fig11.show()
    # fig12.show()

    # Step 5: Save the figures
    output_dir = 'outputs/graphs'
    os.makedirs(output_dir, exist_ok=True)

    pio.write_image(fig1, os.path.join(output_dir, 'messages_over_time.png'))
    pio.write_image(fig2, os.path.join(output_dir, 'message_types_distribution.png'))
    pio.write_image(fig3, os.path.join(output_dir, 'top_actors.png'))
    pio.write_image(fig4, os.path.join(output_dir, 'messages_edited_over_time.png'))
    pio.write_image(fig5, os.path.join(output_dir, 'media_types_distribution.png'))
    pio.write_image(fig6, os.path.join(output_dir, 'messages_per_actor.png'))
    pio.write_image(fig7, os.path.join(output_dir, 'top_media_types.png'))
    pio.write_image(fig8, os.path.join(output_dir, 'messages_by_hour.png'))
    pio.write_image(fig9, os.path.join(output_dir, 'avg_message_length.png'))
    pio.write_image(fig10, os.path.join(output_dir, 'common_words.png'))
    pio.write_image(fig11, os.path.join(output_dir, 'reply_patterns.png'))
    pio.write_image(fig12, os.path.join(output_dir, 'forwarded_messages.png'))

    print(f"Figures saved to {output_dir}")

    # Step 6: Save the DataFrame to CSV
    messages_data.to_csv(os.path.join(output_dir, 'process.csv'), index=False)
    print("DataFrame saved to", os.path.join(output_dir, 'process.csv'))

else:
    print("No data to display")
