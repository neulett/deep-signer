from keypoints_extraction import Models

source_path = input("Source Path : ")
dest_path = input("Dest Path : ")

selected_model = input("Select Model : ")

extractor = Models(source_path, dest_path)
extractor.working_threads(model=selected_model)
