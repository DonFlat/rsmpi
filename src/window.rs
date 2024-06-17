#![allow(missing_docs)]

use std::ffi::{c_int, c_void};
use std::mem::{ManuallyDrop};
use crate::{ffi, Rank};
use crate::topology::UserGroup;
use crate::traits::{AsRaw, Equivalence};

pub struct CreatedWindow<'a, T> where T: Equivalence {
    pub window_vec_ptr: &'a mut Vec<T>,
    pub window_base_ptr: ffi::MPI_Win
}

pub struct AllocatedWindow<T> where T: Equivalence {
    pub window_vector: ManuallyDrop<Vec<T>>,
    pub window_ptr: ffi::MPI_Win
}

pub trait WindowOperations <T> {
    fn get_whole_vector(&mut self, target_rank: usize);
    fn put(
        &mut self,
        origin_disp: usize,
        origin_count: usize,
        target_rank: usize,
        target_disp: usize,
        target_count: usize
    );
    fn put_whole_vector(&mut self, target_rank: usize);
    fn put_from_vector(&mut self, origin: &mut Vec<T>, target_rank: usize);
    fn fence(&self);
    fn start(&self, group: &UserGroup);
    fn complete(&self);
    fn post(&self, group: &UserGroup);
    fn wait(&self);
    fn exclusive_lock(&self, rank: Rank);
    fn unlock(&self, rank: Rank);
}


impl<'a, T> WindowOperations<T> for CreatedWindow<'a, T> where T: Equivalence {
    fn get_whole_vector(&mut self, target_rank: usize) {
        get(self.window_vec_ptr, target_rank, self.window_base_ptr);
    }

    fn put(&mut self, origin_disp: usize, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize) {
        unsafe {
            ffi::MPI_Put(
                self.window_vec_ptr.as_ptr().add(origin_disp) as *mut c_void,
                origin_count as c_int,
                T::equivalent_datatype().as_raw(),
                target_rank as c_int,
                target_disp as ffi::MPI_Aint,
                target_count as c_int,
                T::equivalent_datatype().as_raw(),
                self.window_base_ptr
            );
        }
    }


    fn put_whole_vector(&mut self, target_rank: usize) {
        put(self.window_vec_ptr, target_rank, self.window_base_ptr);
    }

    fn put_from_vector(&mut self, origin: &mut Vec<T>, target_rank: usize) {
        unsafe {
            ffi::MPI_Put(
                origin.as_mut_ptr() as *mut c_void,
                origin.len() as c_int,
                T::equivalent_datatype().as_raw(),
                target_rank as c_int,
                0,
                origin.len() as c_int,
                T::equivalent_datatype().as_raw(),
                self.window_base_ptr
            );
        }
    }

    fn fence(&self) {
        fence(self.window_base_ptr);
    }

    fn start(&self, group: &UserGroup) {
        unsafe {
            ffi::MPI_Win_start(group.as_raw(), 0, self.window_base_ptr);
        }
    }

    fn complete(&self) {
        unsafe {
            ffi::MPI_Win_complete(self.window_base_ptr);
        }
    }

    fn post(&self, group: &UserGroup) {
        unsafe {
            ffi::MPI_Win_post(group.as_raw(), 0, self.window_base_ptr);
        }
    }

    fn wait(&self) {
        unsafe {
            ffi::MPI_Win_wait(self.window_base_ptr);
        }
    }

    fn exclusive_lock(&self, rank: Rank) {
        unsafe {
            ffi::MPI_Win_lock(ffi::MPI_LOCK_EXCLUSIVE as c_int, rank as c_int, 0, self.window_base_ptr);
        }
    }

    fn unlock(&self, rank: Rank) {
        unsafe {
            ffi::MPI_Win_unlock(rank as c_int, self.window_base_ptr);
        }
    }
}

impl<T> WindowOperations<T> for AllocatedWindow<T> where T: Equivalence {
    fn get_whole_vector(&mut self, target_rank: usize) {
        get(&mut self.window_vector, target_rank, self.window_ptr);
    }

    fn put(&mut self, origin_disp: usize, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize) {
        unsafe {
            ffi::MPI_Put(
                self.window_vector.as_ptr().add(origin_disp) as *mut c_void,
                origin_count as c_int,
                T::equivalent_datatype().as_raw(),
                target_rank as c_int,
                target_disp as ffi::MPI_Aint,
                target_count as c_int,
                T::equivalent_datatype().as_raw(),
                self.window_ptr
            );
        }
    }

    fn put_whole_vector(&mut self, target_rank: usize) {
        put(&mut self.window_vector, target_rank, self.window_ptr);
    }

    fn put_from_vector(&mut self, origin: &mut Vec<T>, target_rank: usize) {
        unsafe {
            ffi::MPI_Put(
                origin.as_mut_ptr() as *mut c_void,
                origin.len() as c_int,
                T::equivalent_datatype().as_raw(),
                target_rank as c_int,
                0,
                origin.len() as c_int,
                T::equivalent_datatype().as_raw(),
                self.window_ptr
            );
        }
    }
    fn fence(&self) {
        fence(self.window_ptr);
    }

    fn start(&self, group: &UserGroup) {
        unsafe {
            ffi::MPI_Win_start(group.as_raw(), 0, self.window_ptr);
        }
    }

    fn complete(&self) {
        unsafe {
            ffi::MPI_Win_complete(self.window_ptr);
        }
    }

    fn post(&self, group: &UserGroup) {
        unsafe {
            ffi::MPI_Win_post(group.as_raw(), 0, self.window_ptr);
        }
    }

    fn wait(&self) {
        unsafe {
            ffi::MPI_Win_wait(self.window_ptr);
        }
    }

    fn exclusive_lock(&self, rank: Rank) {
        unsafe {
            ffi::MPI_Win_lock(ffi::MPI_LOCK_EXCLUSIVE as c_int, rank as c_int, 0, self.window_ptr);
        }
    }

    fn unlock(&self, rank: Rank) {
        unsafe {
            ffi::MPI_Win_unlock(rank as c_int, self.window_ptr);
        }
    }
}

pub fn get<T>(vec: &mut Vec<T>, target_rank: usize, window: ffi::MPI_Win) where T: Equivalence {
    unsafe {
        ffi::MPI_Get(
            vec.as_mut_ptr() as *mut c_void,
            vec.len() as c_int,
            T::equivalent_datatype().as_raw(),
            target_rank as c_int,
            0,
            vec.len() as c_int,
            T::equivalent_datatype().as_raw(),
            window
        );
    }
}
pub fn put<T>(vec: &mut Vec<T>, target_rank: usize, window: ffi::MPI_Win) where T: Equivalence {
    unsafe {
        ffi::MPI_Put(
            vec.as_mut_ptr() as *mut c_void,
            vec.len() as c_int,
            T::equivalent_datatype().as_raw(),
            target_rank as c_int,
            0,
            vec.len() as c_int,
            T::equivalent_datatype().as_raw(),
            window
        );
    }
}

fn fence(window: ffi::MPI_Win) {
    unsafe {
        ffi::MPI_Win_fence(0, window);
    }
}

impl<T> Drop for AllocatedWindow<T> where T: Equivalence {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Win_free(&mut self.window_ptr);
        }
    }
}

impl<'a, T> Drop for CreatedWindow<'a, T> where T: Equivalence {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Win_free(&mut self.window_base_ptr);
        }
    }
}