#include "BranchAndCut.h"

orderList *orderList::pasteTask (taskTree *task) {
    orderList *order = new orderList (task);

    if (task->num_of_int > this->task->num_of_int) {
      order->next = this;
      return order;
    }
    orderList *current = this;
    while (current->next != NULL) {
      if (current->next->task->num_of_int < task->num_of_int) {
        break;
      }
      current = current->next;
    }
    order->next = current->next;
    current->next = order;
    return this;
  }
