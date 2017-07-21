/**
 * Copyright (c) 2016 ISP RAS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
