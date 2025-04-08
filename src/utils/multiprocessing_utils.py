def compute_similarity(args):
    """
    Helper function for multiprocessing.
    Args:
        args (tuple): (i, j, query_img, gallery_img, save_logs)
    Returns:
        tuple: (i, j, similarity_score) or (i, j, None) if error occurred.
    """
    i, j, query_img, gallery_img, save_logs = args
    try:
        # Compute the similarity.
        result = query_img.compare_processed_image(gallery_img, save_logs)
        return i, j, result
    except Exception as e:
        print(f"Error computing similarity for query index {i} and gallery index {j}: {e}")
        return i, j, None