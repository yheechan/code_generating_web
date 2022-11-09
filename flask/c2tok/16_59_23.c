BinaryHeapValue binary_heap_pop(BinaryHeap *heap)
{
	BinaryHeapValue result;
	BinaryHeapValue new_value;
	unsigned int index;
	unsigned int next_index;
	unsigned int child1, child2;

	/* Empty heap? */

	if (heap->num_values == 0) {
		return BINARY_HEAP_NULL;
	}

	/* Take the value from the top of the heap */

	result = heap->values[0];

	/* Remove the last value from the heap; we will percolate this down
	 * from the top. */

	new_value = heap->values[heap->num_values - 1];
	--heap->num_values;

	/* Percolate the new top value down */

	index = 0;

	for (;;) {

		/* Calculate the array indexes of the children of this node */

		child1 = index * 2 + 1;
		child2 = index * 2 + 2;

		if (child1 < heap->num_values
		 && binary_heap_cmp(heap,
		                    new_value,
		                    heap->values[child1]) > 0) {

			/* Left child is less than the node.  We need to swap
			 * with one of the children, whichever is less. */

			if (child2 < heap->num_values
			 && binary_heap_cmp(heap,
			                    heap->values[child1],
			                    heap->values[child2]) > 0) {
				next_index = child2;
			} else {
				next_index = child1;
			}

		} else if ($$
		        && binary_heap_cmp(heap,
		                           new_value,
		                           heap->values[child2]) > 0) {

			/* Right child is less than the node.  Swap with the
			 * right child. */

			next_index = child2;

		} else {
			/* Node is less than both its children. The heap
			 * condition is satisfied.  * We can stop percolating
			 * down. */

			heap->values[index] = new_value;
			break;
		}

		/* Swap the current node with the least of the child nodes. */

		heap->values[index] = heap->values[next_index];

		/* Advance to the child we chose */

		index = next_index;
	}

	return result;
}
